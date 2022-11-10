import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy

from sklearn.base import BaseEstimator
from collections import defaultdict

from .tokenizer import Tokenizer
from .FocalLoss import FocalLoss
from .EmbbedingTFIDF import EmbbedingTFIDF
from .Similarities import SimMatrix, DistMatrix
from ...metrics.tick import Tick

class AttentionTFIDF(nn.Module):
    def __init__(self, vocab_size: int, nclass: int, hiddens: int=300, maxF: int=20, drop: float = .5,
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', 
                att_model: str ='AA', sim_func='dist', norep=2):
        super(AttentionTFIDF, self).__init__()
        #self.fc        = nn.Linear(hiddens, nclass+1)
        self.fc       = nn.Sequential( nn.Linear(hiddens, nclass+norep), nn.Softmax(dim=-1))
        self.drop_    = nn.Dropout(drop)
        self.emb_     = EmbbedingTFIDF(vocab_size, hiddens, maxF=20, drop=self.drop_,  att_model=att_model)
        self.loss_f   = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.nclass   = nclass
        self.bias     = nn.parameter.Parameter(torch.zeros(nclass+norep))
        if sim_func == 'dist':
            self.sim_func = DistMatrix()
        elif sim_func == 'sim':
            self.sim_func = SimMatrix()
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc[0].weight.data)
        
    def forward(self, doc_tids, TFs, DFs, labels=None):
        
        result = self.emb_(doc_tids, TFs, DFs)
        
        co_weights  = self.sim_func( result['K'], result['Q'] ) # SIMILARITY(Q:[B,L,D], Q:[B,L,D]) -> W:[B,L,L]
        co_weights[result['pad_mask'].logical_not()] = 0.
        
        V = torch.softmax(co_weights, dim=-1) @ result['V']    # W:[B,L,L] @ V:[B,L,D] -> V':[B,L,D]
        V = self.drop_(V)
        V_lgts = self.fc(V)                                    # FC(V':[B,L,D])-> V_lgs:[B,L,C+b]
        
        weights = co_weights.sum(axis=-2)                       # sum(co_weights:[B,L,L], 1) -> weights:[B,L]
        weights = weights / result['doc_sizes']                 # weights:[B,L] / d_sizes:[B,1]
        weights[result['bx_packed']] = float('-inf')
        weights = torch.softmax(weights, dim=-1)                 # softmax(weights:[B,L]) -> weights:[B,L]
        
        zeros   = torch.zeros_like(weights)
        isnan   = torch.isnan(weights)
        weights = torch.where(isnan, zeros, weights)
        
        weights = weights.view( *weights.shape, 1 )            # weights:[B,L] -> weights:[B,L,1]
        
        V_lgts = V_lgts * weights                              # V_lgs:[B,L,C+1] * weights:[B,L,1] -> logits:[B,L,C+1]
        
        logits = V_lgts.sum(dim=1)[:,:self.nclass]             # V_lgts:logits:[B,L,C+b] -> logits:[B,C]
        logits = torch.softmax(logits, dim=-1)                 # softmax(logits:[B,C]) -> logits:[B,C]
        
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

class AttTFIDFClassifier(BaseEstimator):
    def __init__(self, tknz = {}, model = {}, nepochs:int=50,
                max_drop:float=.75, batch_size:int=16,
                weight_decay:float = 5e-3, lr:float = 5e-3, device='cuda'):
        super(AttTFIDFClassifier, self).__init__()
        self.model      = model
        self.tknz       = tknz
        if isinstance(self.tknz, dict):
            self.tknz = Tokenizer(**self.tknz)

        self.device       = device
        self.batch_size   = batch_size
        self.max_drop     = max_drop
        self.nepochs      = nepochs
        self.weight_decay = weight_decay
        self.lr           = lr
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        statatistics_ = defaultdict(Tick)
        if not self.tknz.is_fit:
            self.tknz.fit( X_train, y_train )
        
        if isinstance(self.model, dict):
            self.model["vocab_size"] = self.tknz.vocab_max_size
            self.model["nclass"]     = self.tknz.n_class
            self.model               = AttentionTFIDF(**self.model)
        self.model     = self.model.to(self.device)
        self.optimizer = AdamW( self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=.95, patience=3, verbose=True)
        best = 99999.
        best_acc = 0.
        counter = 1
        dl_val = DataLoader(list(zip(X_val, y_val)), batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.tknz.collate_val)

        for e in tqdm(range(self.nepochs), total=self.nepochs):
            dl_train = DataLoader(list(zip(enumerate(X_train), y_train)), batch_size=self.batch_size, shuffle=True, collate_fn=self.tknz.collate_train)
            with tqdm(total=len(y_train)+len(y_val), smoothing=0., desc=f"V-ACC={best_acc:.3} L={best:.6} E={e+1}") as pbar:
                loss_train  = 0.
                total = 0.
                correct  = 0.
                self.model.train()
                y_true = []
                y_preds = []
                for i, data in enumerate(dl_train):
                    data = { k: v.to(self.device) for (k,v) in data.items() }
                    data.pop('didxs')

                    result = self.model( **data )
                    loss   = result['loss']

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_train    += result['loss'].item()

                    y_pred         = result['logits'].argmax(axis=-1)
                    correct       += (y_pred == data['labels']).sum().item()
                    total         += len(data['labels'])

                    y_true.extend(list(data['labels'].cpu()))
                    y_preds.extend(list(y_pred.cpu()))

                    self.model.drop_.p  = (correct/total)*self.max_drop
                    print(f"t-ACC: {(correct/total):.3} Drp: {self.model.drop_.p:.3} L={(loss_train/(i+1)):.6} iter={i+1}", end=f"{ ''.join([' ']*100) }\r")
                    pbar.update( len(data['labels']) )
                    del result, data

                f1_ma = f1_score(y_true, y_preds, average='macro')*100.
                f1_mi = f1_score(y_true, y_preds, average='micro')*100.
                print(f"--F1: ({f1_mi:.3}/{f1_ma:.3}) Drp: {self.model.drop_.p:.3} L={(loss_train/(i+1)):.6}{ ''.join([' ']*100) }")
                loss_train = loss_train/(i+1)
                total = 0.
                correct  = 0.
                loss_val = 0.
                self.model.eval()
                y_true  = []
                y_preds = [] 
                for i, data in enumerate(dl_val):
                    data = { k: v.to(self.device) for (k,v) in data.items() }
                    result = self.model( **data )

                    loss_val   += result['loss'].item()
                    y_pred      = result['logits'].argmax(axis=-1)
                    correct    += (y_pred == data['labels']).sum().item()
                    total      += len(data['labels'])
                    pbar.update( len(data['labels']) )

                    y_true.extend(list(data['labels'].cpu()))
                    y_preds.extend(list(y_pred.cpu()))

                    del result, data
                f1_ma = f1_score(y_true, y_preds, average='macro')
                f1_mi = f1_score(y_true, y_preds, average='micro')
                print(f"v-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M=", end="")
                loss_val   = (loss_val/(i+1)) / ( f1_ma + f1_mi )
                print(f"{loss_val:.5}")
                self.scheduler.step(loss_val)

                if best-loss_val > 0.0001 :
                    best = loss_val
                    counter = 1
                    best_acc = correct/total
                    best_model = copy.deepcopy(self.model).to('cpu')
                    print('*')
                elif counter > 10:
                    break
                else:
                    counter += 1
        self.model = best_model.to(self.device)
        return statatistics_
        
    def predict(self, X):
        dl_test = DataLoader(X, batch_size=self.batch_size*2, shuffle=False, collate_fn=self.tknz.collate)
        self.model.eval()
        y_preds = []
        for i, data in tqdm(enumerate(dl_test)):
            data = { k: v.to(self.device) for (k,v) in data.items() }
            result = self.model( **data )
            y_preds.extend(result['logits'].argmax(axis=-1).long().cpu().tolist())
        return self.tknz.le.inverse_transform(y_preds)