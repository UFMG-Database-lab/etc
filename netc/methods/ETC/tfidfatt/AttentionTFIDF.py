import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
from numpy import ceil

from sklearn.base import BaseEstimator
from collections import defaultdict

from ..FocalLoss import FocalLoss
from ..EmbbedingTFIDF import EmbbedingTFIDF
from ..Similarities import SimMatrix, DistMatrix

class AttentionTFIDF(nn.Module):
    def __init__(self, vocab_size: int, hiddens: int, nclass: int, maxF: int=20, nheads: int=6,
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', drop: float = .5,
                att_model: str ='AA', sim_func='dist', norep=2):
        super(AttentionTFIDF, self).__init__()
        self.D   = hiddens    # number of   (D)imensions
        self.C   = nclass     # number of   (C)lass
        self.H   = nheads     # number of   (H)eads on multhead
        self.V   = vocab_size # size of the (V)ocabulary
        self.P   = norep
        #self.norm     = nn.LayerNorm(self.H)
        self.norm     = nn.BatchNorm1d(self.H)
        self.fc       = nn.Sequential( nn.Linear(self.D, self.C+self.P), nn.Softmax(dim=-1))
        self.drop_    = nn.Dropout(drop)
        self.emb_     = EmbbedingTFIDF(vocab_size, self.D, maxF=maxF, drop=self.drop_,  att_model=att_model)
        self.loss_f   = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.dist_func = DistMatrix()
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc[0].weight.data)
        
    def getHidden(self, hidd):
        B, L, D = hidd.shape
        assert D == self.D
        hidd = hidd.view( B, L, self.H, self.D//self.H )
        hidd = hidd.transpose(1,2)
        return hidd
    
    def catHiddens(self, hidd):
        B, H, L, D = hidd.shape
        assert H*D == self.D
        assert H == self.H
        hidd = hidd.transpose(1,2)
        hidd = hidd.reshape(B, L, H*D)
        return hidd
    def removeNaN(self, data):
        zeros   = torch.zeros_like(data)
        isnan   = torch.isnan(data)
        return torch.where(isnan, zeros, data)
    def lnormalize(self, co_weights):
        B,H,L,_ = co_weights.shape
        nco_weights = co_weights.view(B,H,L*L).transpose(-1,-2)
        nco_weights = self.norm(nco_weights)
        return nco_weights.transpose(-1,-2).view(B,H,L,L)
    def bnormalize(self, co_weights):
        B,H,L,_ = co_weights.shape
        nco_weights = co_weights.reshape(B,H,L*L)
        nco_weights = self.norm(nco_weights)
        return nco_weights.reshape(B,H,L,L)
    
    def forward(self, doc_tids, TFs, DFs, labels=None):
        
        result = self.emb_(doc_tids, TFs, DFs) 
        K = self.getHidden(result['K']) # K:[B,L,D] -> L:[B,H,L,D//H]
        Q = self.getHidden(result['Q']) # Q:[B,L,D] -> Q:[B,H,L,D//H]
        B,H,L,_ = Q.shape
        
        co_weights  = self.dist_func( K, Q ) # SIMILARITY(Q:[B,H,L,D//H], Q:[B,H,L,D//H]) -> W:[B,H,L,L]
        co_weights  = self.bnormalize(co_weights)
        
        pad_mask = result['pad_mask'].unsqueeze(1).repeat([1, H, 1, 1]) # pm:[B, L, L] -> pm:[B, H, L, L]
        co_weights[pad_mask.logical_not()] = 0.
        co_weights  = torch.softmax(co_weights, dim=-1)                     # co:[B, H, L, L]
        co_weights = self.removeNaN(co_weights)
        
        V = self.getHidden(result['V'])
        V = co_weights @ V    # W:[B,H,L,L] @ V:[B,H,L,D//H] -> V':[B,H,L,D//H]
        V = self.drop_(V)
        V_lgts = self.fc(self.catHiddens(V))         # FC(V':[B,L,D])-> V_lgs:[B,L,C+b]
        
        weights = co_weights.sum(axis=-2).mean(axis=-2)    # sum(co_weights:[B,H,L,L], -2) -> weights:[B,L]

        weights = weights / result['doc_sizes']                   # weights:[B,L] / d_sizes:[B,1]
        #bx_pack = result['bx_packed'].unsqueeze(1).repeat([1,self.H,1])
        weights[result['bx_packed']] = float('-inf')
        weights = torch.softmax(weights, dim=-1)     # softmax(weights:[B,L]) -> weights:[B,L]
        weights = self.removeNaN(weights).unsqueeze(-1)  # weights:[B,L] -> weights:[B,L,1]
        
        V_lgts = V_lgts * weights                        # V_lgs:[B,L,C+1] * weights:[B,L,1] -> logits:[B,L,C+1]

        logits = V_lgts.sum(dim=-2)[:,:self.C]         # V_lgts:logits:[B,L,C+b] -> logits:[B,C]
        logits = torch.softmax(logits, dim=-1)           # softmax(logits:[B,C]) -> logits:[B,C]
         
        result_ = { 't_probs': V_lgts, 'logits': logits}
        if labels is not None:
            result_['loss'] = self.loss_f(logits, labels)
        
        return result_

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
