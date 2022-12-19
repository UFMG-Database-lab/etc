from ..EmbbedingTFIDF import EmbbedingTFIDF
from ..FocalLoss import FocalLoss
from ..Similarities import DistMatrix

from torch import nn
import torch

class nearAttention(nn.Module):
    def __init__(self, hiddens: int, nheads: int=6):
        super(nearAttention, self).__init__()
        self.H = nheads
        self.D = hiddens
        self.dist_func = DistMatrix()
        self.norm1     = nn.BatchNorm1d(self.H)
        self.norm2     = nn.BatchNorm1d(self.H)
        self.posterior = nn.Sequential( nn.Linear(self.H, 2), nn.Softmax(dim=-1) )
        
    def getHidden(self, hidd):
        B, L, D = hidd.shape
        assert D == self.D
        hidd = hidd.view( B, L, self.H, self.D//self.H )
        hidd = hidd.transpose(1,2)
        return hidd
    def removeNaN(self, data):
        zeros   = torch.zeros_like(data)
        isnan   = torch.isnan(data)
        return torch.where(isnan, zeros, data)
    def forward(self, Q, K, V, pad_mask, bx_packed, **kargs):
        Q = self.getHidden(Q) # Q:[B,L,D] -> Q:[B,H,L,d]
        K = self.getHidden(K) # K:[B,L,D] -> K:[B,H,L,d]
        V = self.getHidden(V) # V:[B,L,D] -> V:[B,H,L,d]
        
        B,H,L,_ = Q.shape
        pad_mask = pad_mask.unsqueeze(1).repeat([1, H, 1, 1]).logical_not() # pm:[B, L, L] -> pm:[B, H, L, L]
        
        # co_weights (cW)
        co_weights = self.dist_func( K, Q )              # SIMILARITY(Q:[B,H,L,D//H], Q:[B,H,L,D//H]) -> cW:[B,H,L,L] 
        co_weights = co_weights.reshape(B,H,L*L)         # cW:[B,H,L,L] -> cW:[B,H,L*L]
        co_weights = self.norm1(co_weights)              # batchNorm(cW:[B,H,L*L]) -> cW:[B,H,L*L]
        co_weights = co_weights.reshape(B,H,L,L)         # cW:[B,H,L*L] -> cW:[B,H,L,L]
        co_weights[pad_mask] = 0.                        # fill(cW:[B,H,L,L], pad)
        co_weights = torch.softmax(co_weights, dim=-1)   # softmax(cW:[B, H, L, L]) -> cW:[B, H, L, L']
        co_weights = self.removeNaN(co_weights)          # fill(cW:[B,H,L,L'], NaN)
        
        # weights (w)
        weights = co_weights.sum(axis=-2)                # sum(cW:[B,H,(L),L], -2) -> w:[B,H,L]
        weights = self.norm2(weights).transpose(1,2)     # batchNorm(w:[B,H,L]).T -> w:[B,L,H]
        weights = self.posterior(weights)                # P(w:[B,L,H]|C) -> w:[B,L,2]
        weights = weights[:,:,0].squeeze(-1)             # w:[B,L,2] -> w:[B,L,1] -> w:[B,L]
        weights = self.removeNaN(weights)                # fill(w:[B,L,1], NaN)
        weights[bx_packed] = float('-inf')               # fill(w:[B,L,1], packed)
        weights = torch.softmax(weights, dim=-1)         # softmax(w:[B,L]) -> w:[B,L]
        weights = self.removeNaN(weights).unsqueeze(-1)  # fill(w:[B,L], NaN) -> w:[B,L] -> w:[B,L,1]
        
        return { 'co_weights': co_weights, 'weights': weights, 'V': V } # 
class EnsembleTC(nn.Module):
    def __init__(self, drop, hiddens: int, nclass:int, norep:int = 2, nheads: int=6, dev: bool=False):
        super(EnsembleTC, self).__init__()
        self.H    = nheads
        self.D    = hiddens
        self.C    = nclass
        self.P    = norep
        self._dev = dev
        self.fc   = nn.Sequential( nn.Linear(self.D, self.C+self.P), nn.Softmax(dim=-1))
        
    def catHiddens(self, hidd):
        B, H, L, d = hidd.shape
        assert H*d == self.D
        assert H == self.H
        hidd = hidd.transpose(1,2)
        hidd = hidd.reshape(B, L, H*d)
        return hidd
    def forward(self, V, co_weights, weights):
        if self._dev:
            old_V_lgts = self.fc(self.catHiddens(V)) * weights
        V = co_weights @ V                                      # W:[B,H,L,L] @ V:[B,H,L,d] -> V':[B,H,L,d]
        V = self.catHiddens(V)                                  # V':[B,H,L,d] -> V':[B,L,D]
        V_lgts = self.fc(V)                                     # V':[B,H,L,d] -> P:[B,L,C+P]
        V_lgts = V_lgts * weights                               # P:[B,L,C+P] * w:[B,H,L] -> P:[B,L,C+P]
        logits = V_lgts.sum(dim=-2)[:,:self.C]                  # logits:[B,L,C+P] -> logits:[B,C+P] -> logits:[B,C]
        logits = torch.softmax(logits, dim=-1)                  # softmax(logits:[B,C]) -> logits:[B,C]
        
        returing = { 'logits': logits, 'V_logits': V_lgts, 'V': V }
        if self._dev:
            returing['old_V_lgts'] = old_V_lgts
        
        return returing
class ETCModel(nn.Module):
    def __init__(self, vocab_size: int, hiddens: int, nclass: int, maxF: int=20, nheads: int=6,
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', drop: float = .5,
                att_model: str ='AA', norep=2, dev=False):
        super(ETCModel, self).__init__()
        self.D    = hiddens    # number of   (D)imensions
        self.C    = nclass     # number of   (C)lass
        self.H    = nheads     # number of   (H)eads on multihead
        self.V    = vocab_size # size of the (V)ocabulary
        self.P    = norep      # number of   (P)riors
        self._dev  = dev
        self.drop_ = nn.Dropout(drop)
        
        self.emb_  = EmbbedingTFIDF(self.V, self.D, maxF=maxF, drop=self.drop_,  att_model=att_model)
        self.nAtt_ = nearAttention(self.D, self.H)
        self.etc_  = EnsembleTC(drop=self.drop_, hiddens=self.D, nclass=self.C, norep=self.P, nheads=self.H, dev=self._dev)
        self.loss_ = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
    
    def forward(self, doc_tids, TFs, DFs, labels=None):
        
        emb  = self.emb_(doc_tids, TFs, DFs) 
        att  = self.nAtt_(**emb)
        ensb = self.etc_(**att)
        
        result = { 't_probs': ensb['V_logits'], 'logits': ensb['logits'], 'V': ensb['V']}
        if labels is not None:
            result['loss'] = self.loss_(result['logits'], labels)
        if self._dev:
            result['old_V'] = emb['V']
            result['co_weights'] = att['co_weights']
            result['weights'] = att['weights']
            result['old_V_lgts'] = ensb['old_V_lgts']
        
        return result
