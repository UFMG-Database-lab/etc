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
    def __init__(self, hiddens: int, drop_: nn.Dropout, nheads: int=6):
        super(nearAttention, self).__init__()
        self.H = nheads
        self.D = hiddens
        self.d = hiddens // nheads
        self.drop_ = drop_
        self.dist      = DistMatrix()
        self.q_norm    = nn.BatchNorm2d(self.H, self.d)
        self.k_norm    = nn.BatchNorm2d(self.H, self.d)
        self.dQK_norm  = nn.Sequential(nn.BatchNorm1d(self.H), nn.LeakyReLU(negative_slope=99.))
    def forward(self, Q, K, V, pad_mask, **kargs):
        B,H,L,d  = Q.shape
        
        # co_weights (cW)
        co_weights = self.dist( K, Q )                   # distL2(Q:[B,H,L,D//H], Q:[B,H,L,D//H]) -> cW:[B,H,L,L] 
        co_weights = co_weights.reshape(B,H,L*L)         # cW:[B,H,L,L] -> cW:[B,H,L*L]
        co_weights = self.dQK_norm(co_weights)           # batchNorm(cW:[B,H,L*L]) -> cW:[B,H,L*L]
        co_weights = co_weights.reshape(B,H,L,L)         # cW:[B,H,L*L] -> cW:[B,H,L,L]
        if 'co_weights' in kargs:
            co_weights = co_weights + kargs['co_weights']
        co_weights[pad_mask] = float('-inf')             # fill(cW:[B,H,L,L], pad)
        co_weights = torch.softmax(co_weights, dim=-1)   # softmax(cW:[B, H, L, L]) -> cW:[B, H, L, L']
        co_weights = removeNaN(co_weights)               # fill(cW:[B,H,L,L'], NaN)
        
        kargs['co_weights'] = co_weights
        kargs['pad_mask'] = pad_mask
        kargs['V'] = co_weights @ V
        kargs['K'] = self.k_norm( (co_weights @ K).transpose(-1,-2) ).transpose(-1,-2)
        kargs['Q'] = self.q_norm( (co_weights @ Q).transpose(-1,-2) ).transpose(-1,-2)
        
        return kargs # 
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
        self.wei_norm = nn.Sequential( nn.BatchNorm1d(self.H), nn.Softmax(dim=-1) )
        
        self.emb_  = EmbbedingTFIDF(self.V, self.D, maxF=maxF, drop=self.drop_,  att_model=att_model)
        self.nAtt_ = nn.Sequential(*[nearAttention(self.D, self.drop_, self.H) for _ in range(self.la) ])
        self.loss_s = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        
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
        
        
        W = att['co_weights']           # [B,H,L,L']
        W = W.sum(dim=-2).mean(dim=-2, keepdims=True)     # [B,1,L]
        W = W.transpose(1,2)                             # [B,L,1]
        W[att['bx_packed']] = float('-inf')
        W = torch.softmax(W, dim=1)
        W = removeNaN(W)
        
        V      = self.catHiddens(att['V']) # [B,L,D]
        
        V_lgts = (W * self.fc(V)).sum(dim=1)      # [B,L,1]*[B,L,P+C] -> [B,P+C]
        V_lgts = torch.softmax(V_lgts, dim=-1)    # [B,P+C]
        EV     = -torch.log2( torch.pow(V_lgts, V_lgts) ).sum(dim=-1, keepdims=True) # H(P) = -sum_i log2( P_i^P_i ) = -sum_i P_i * log2(P_i)
        
        M_lgts = self.fc((W * V).sum(dim=1))
        M_lgts = torch.softmax(M_lgts, dim=-1)
        EM     = -torch.log2( torch.pow(M_lgts, M_lgts) ).sum(dim=-1, keepdims=True)
        
        E      = EM+EV
        
        logits = (EV/E)*V_lgts + (EM/E)*M_lgts # [B,L,1] * [B,L,D] -> [B,D] -> [B, P+C]
        logits = torch.softmax(logits[:,self.P:], dim=-1)
        
        result = { 'logits': logits, 'V': V}
        if labels is not None:
            result['loss'] = self.loss_s(logits, labels) # FLoss(logits:[B,C]; labels:[B]) -> loss1: 1
        
        return result