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

from .ETC.tokenizer import Tokenizer
from .ETC.EnsembleTC.ETCModel import ETCModel
from ..metrics.tick import Tick


class ETCClassifier(BaseEstimator):
    def __init__(self, tknz = {}, model = {}, nepochs:int=50,
                max_drop:float=.75, batch_size:int=16, min_f1=.97,
                weight_decay:float = 5e-3, lr:float = 5e-3, device='cuda'):
        super(ETCClassifier, self).__init__()
        self.model      = model
        self.tknz       = tknz
        self.min_f1     = min_f1
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
            self.model               = ETCModel(**self.model)
        self.model     = self.model.to(self.device)
        self.optimizer = AdamW( self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=.95, patience=3, verbose=True)
        best = 99999.
        trained_f1 = (0,0)
        counter = 1
        dl_val = DataLoader(list(zip(X_val, y_val)), batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.tknz.collate_val)
        with tqdm(total=self.nepochs, position=3, desc="First epoch") as e_pbar:
            with tqdm(total=len(y_train)+len(y_val), position=4, smoothing=0., desc=f"First batch") as b_pbar:
                b_pbar.reset(total=len(y_train)+len(y_val))
                for e in range(self.nepochs):
                    dl_train = DataLoader(list(zip(enumerate(X_train), y_train)),
                                            batch_size=self.batch_size, shuffle=True,
                                            collate_fn=self.tknz.collate_train)
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
                        b_pbar.desc = f"--ACC: {(correct/total):.3} ({trained_f1[0]:.2},{trained_f1[1]:.2}) L={(loss_train/(i+1)):.6} b={i+1}"
                        b_pbar.update( len(data['labels']) )
                        del result, data

                    f1_ma = f1_score(y_true, y_preds, average='macro')
                    f1_mi = f1_score(y_true, y_preds, average='micro')
                    trained_f1 = (f1_mi, f1_ma)
                    b_pbar.desc = f"t-F1: ({f1_mi:.3}/{f1_ma:.3}) L={(loss_train/(i+1)):.6}"
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
                        b_pbar.update( len(data['labels']) )

                        y_true.extend(list(data['labels'].cpu()))
                        y_preds.extend(list(y_pred.cpu()))

                        del result, data
                    f1_ma  = f1_score(y_true, y_preds, average='macro')
                    f1_mi  = f1_score(y_true, y_preds, average='micro')
                    metric = (loss_val/(i+1)) / ( f1_ma + f1_mi )
                    self.scheduler.step(loss_val)

                    if best-metric > 0.0001:
                        best = metric
                        counter = 1
                        best_acc = correct/total
                        best_model = copy.deepcopy(self.model).to('cpu')
                        b_pbar.desc = f"*-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}"
                        e_pbar.desc = f"v-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}"
                    elif counter > 10:
                        break
                    elif trained_f1[0] > self.min_f1 and trained_f1[1] > self.min_f1:
                        counter += 1
                    e_pbar.update(1)
                    b_pbar.update(-(len(y_train)+len(y_val)))
        self.model = best_model.to(self.device)
        return statatistics_
        
    def predict(self, X):
        dl_test = DataLoader(X, batch_size=self.batch_size*2, shuffle=False, collate_fn=self.tknz.collate)
        self.model.eval()
        y_preds = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dl_test)):
                data = { k: v.to(self.device) for (k,v) in data.items() }
                result = self.model( **data )
                y_preds.extend(result['logits'].argmax(axis=-1).long().cpu().tolist())
        return self.tknz.le.inverse_transform(y_preds)


    # implement a BatchNorm1dCDF class cdf of Normal distribution (torch.nn.distributions.Normal) considering the running_mean and running_var and as registered buffer