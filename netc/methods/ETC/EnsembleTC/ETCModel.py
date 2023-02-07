from ..EmbbedingTFIDF import EmbbedingTFIDF
from ..FocalLoss import FocalLoss
from ..Similarities import DistMatrix

from torch import nn
import torch

def removeNaN(data):
    zeros   = torch.zeros_like(data)
    isnan   = torch.isnan(data)
    return torch.where(isnan, zeros, data)
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
    def forward(self, V, co_weights, bx_packed):
        
        # weights (w)
        weights = co_weights.sum(axis=-2)                # sum(cW:[B,H,(L),L], -2) -> w:[B,H,L]
        #weights = self.norm(weights)                     # batchNorm(w:[B,H,L]) -> w:[B,H,L]
        #weights = weights.transpose(1,2)                 # w:[B,H,L] -> w:[B,L,H]
        weights = weights.mean(dim=-2)                   # mean(w:[B,H,L]) -> w:[B,L]
        doc_sizes = bx_packed.logical_not().sum().unsqueeze(-1)
        weights = weights / doc_sizes
        weights[bx_packed] = float('-inf')               # fill(w:[B,L], packed)
        weights = torch.softmax(weights, dim=-1)         # softmax(w:[B,L]) -> w:[B,L]
        weights = removeNaN(weights).unsqueeze(-1)       # fill(w:[B,L], NaN) -> w:[B,L] -> w:[B,L,1]
        
        if self._dev:
            old_V_lgts = self.fc(self.catHiddens(V)) * weights
        V = co_weights @ V                                      # W:[B,H,L,L] @ V:[B,H,L,d] -> V':[B,H,L,d]
        V = self.catHiddens(V)                                  # V':[B,H,L,d] -> V':[B,L,D]
        V_lgts = self.fc(V)                                     # V':[B,H,L,d] -> P:[B,L,C+P]float('-inf')
        V_lgts = V_lgts * weights                               # P:[B,L,C+P] * w:[B,H,L] -> P:[B,L,C+P]
        logits = V_lgts.sum(dim=-2)[:,:self.C]                  # logits:[B,L,C+P] -> logits:[B,C+P] -> logits:[B,C]
        logits = torch.softmax(logits, dim=-1)                  # softmax(logits:[B,C]) -> logits:[B,C]
        
        returing = { 'logits': logits, 'V_logits': V_lgts, 'V': V }
        if self._dev:
            returing['old_V_lgts'] = old_V_lgts
            returing['weights'] = weights
        
        return returing

class nearAttention(nn.Module):
    def __init__(self, hiddens: int, nheads: int=6):
        super(nearAttention, self).__init__()
        self.H = nheads
        self.D = hiddens
        self.d = hiddens // nheads
        self.dist      = DistMatrix()
        self.V_norm    = nn.BatchNorm2d(self.H, self.d, affine=False)
        self.Q_norm    = nn.BatchNorm2d(self.H, self.d, affine=False)
        self.K_norm    = nn.BatchNorm2d(self.H, self.d, affine=False)
        self.dQK_norm  = nn.Sequential(nn.BatchNorm1d(self.H), nn.LeakyReLU(negative_slope=9.))
    def forward(self, Q, K, V, pad_mask, **kargs):
        #V = V.transpose(-1,-2) # V:[B,H,L,d] -> V:[B,H,d,L]
        #V = self.V_norm( V )   # V:[B,H,d,L] -> V:[B,H,d,L]
        #V = V.transpose(-1,-2) # V:[B,H,d,L] -> V:[B,H,L,d]
        V = self.V_norm(V.transpose(-1,-2)).transpose(-1,-2)
        
        B,H,L,_  = Q.shape
        
        # co_weights (cW)
        co_weights = self.dist( K, Q )                   # distL2(Q:[B,H,L,D//H], Q:[B,H,L,D//H]) -> cW:[B,H,L,L] 
        co_weights = co_weights.reshape(B,H,L*L)         # cW:[B,H,L,L] -> cW:[B,H,L*L]
        co_weights = self.dQK_norm(co_weights)           # batchNorm(cW:[B,H,L*L]) -> cW:[B,H,L*L]
        co_weights = co_weights.reshape(B,H,L,L)         # cW:[B,H,L*L] -> cW:[B,H,L,L]
        co_weights[pad_mask] = 0.                        # fill(cW:[B,H,L,L], pad)
        co_weights = torch.softmax(co_weights, dim=-1)   # softmax(cW:[B, H, L, L]) -> cW:[B, H, L, L']
        co_weights = removeNaN(co_weights)               # fill(cW:[B,H,L,L'], NaN)
        
        V = co_weights @ V
        Q = co_weights @ Q
        Q = self.Q_norm(Q.transpose(-1,-2)).transpose(-1,-2)
        K = co_weights @ K
        K = self.K_norm(K.transpose(-1,-2)).transpose(-1,-2)
        
        return { 'co_weights': co_weights, 'V': V, 'Q': Q, 'K': K, 'pad_mask': pad_mask } # 

class ETCModel(nn.Module):
    def __init__(self, vocab_size: int, hiddens: int, nclass: int, maxF: int=20, nheads: int=6,
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', drop: float = .5,
                att_model: str ='AA', layers=1, norep=1, dev=False):
        super(ETCModel, self).__init__()
        self.D    = hiddens    # number of   (D)imensions
        self.C    = nclass     # number of   (C)lass
        self.H    = nheads     # number of   (H)eads on multihead
        self.d    = self.D // self.H
        self.V    = vocab_size # size of the (V)ocabulary
        self.P    = norep      # number of   (P)riors
        self.la   = layers
        self._dev  = dev
        self.drop_ = nn.Dropout(drop)
        self.wei_norm = nn.Sequential( nn.Linear(self.H, 2), nn.Softmax(dim=-1) )
        
        self.emb_  = EmbbedingTFIDF(self.V, self.D, maxF=maxF, drop=self.drop_,  att_model=att_model)
        self.nAtt_ = nn.Sequential(*[nearAttention(self.D, self.H) for _ in range(self.la) ])
        self.loss_m = FocalLoss(gamma=gamma, alpha=alpha, reduction='mean')
        self.loss_s = FocalLoss(gamma=gamma, alpha=alpha, reduction='sum')
        
        self.fc     = nn.Sequential(nn.Linear(self.D, self.C+self.P), nn.Softmax(dim=-1))
        
    def getHidden(self, hidd):
        B,L,_ = hidd.shape
        hidd  = hidd.view( B, L, self.H, self.d )
        hidd  = hidd.transpose(1,2)
        return hidd
    
    def catHiddens(self, hidd):
        B, H, L, d = hidd.shape
        assert H*d == self.D
        assert H == self.H
        hidd = hidd.transpose(1,2)
        hidd = hidd.reshape(B, L, H*d)
        return hidd
    
    def forward(self, doc_tids, TFs, DFs, labels=None):
        att  = self.emb_(doc_tids, TFs, DFs) 
        att["pad_mask"] = att["pad_mask"].unsqueeze(1).repeat([1, self.H, 1, 1]).logical_not() # pm:[B, L, L] -> pm:[B, H, L, L]
        att  = { k: self.getHidden(v) if k in ('K', 'Q', 'V') else v for (k,v) in att.items() }
        for nAttLayer in self.nAtt_:
            att  = nAttLayer(**att)
        
        V      = self.catHiddens(att['V'])
        V_lgts = self.fc(V)                # [B, L, C+P]
        
        co_wei = att['co_weights']                     # [B,H,L,L']
        B, H, L,_ = co_wei.shape
        co_wei = co_wei.transpose(-1,-2)               # [B,L,L',H]
        co_wei = co_wei.reshape(B, L*L, H)             # [B,L*L,2]
        co_wei = self.wei_norm(co_wei)                 # [B,L*L,2]
        co_wei = co_wei.reshape(B, L, L, 2)            # [B,L,L,2]
        co_wei = co_wei.sum(dim=-2, keepdims=True)     # [B,L,1,2]
        co_wei = co_wei[:,:,:,0]                       # [B,L,1]
        co_wei = torch.softmax(co_wei, dim=-1)
        
        
        logits = (co_wei * V_lgts).sum(dim=1)
        logits = logits[:,:self.C]
        logits = torch.softmax(logits, dim=-1)
        
        result = { 'logits': logits, 'V': V}
        if labels is not None:
            B,L,C   = V_lgts.shape
            result['loss'] = 0.
            result['loss'] += self.loss_s(logits, labels) # FLoss(logits:[B,C]; labels:[B]) -> loss1: 1
            #result['loss'] += self.loss_m(V_lgts.reshape( B*L,C ), labels.repeat( L ))  # FLoss(V_lgts:[B*L,C]; labels:[B*L]) -> loss1: 1
        
        return result