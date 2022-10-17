import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import numpy as np
import copy

from .tokenizer import Tokenizer
from .FocalLoss import FocalLoss
from .EmbbedingTFIDF import EmbbedingTFIDF
from .Similarities import SimMatrix, DistMatrix
from sklearn.base import BaseEstimator


class AttentionTFIDF(nn.Module):
    def __init__(self, vocab_size: int, hiddens: int, nclass: int, maxF: int=20, drop: float = .75,
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', negative_slope=9.,
                att_model: str ='AA', sim_func='D'):
        super(AttentionTFIDF, self).__init__()
        self.fc     = nn.Sequential(nn.Linear(hiddens, nclass), nn.Softmax(dim=-1))
        self.drop_  = nn.Dropout(drop)
        self.emb_   = EmbbedingTFIDF(vocab_size, hiddens, maxF=maxF, drop=self.drop_,  att_model=att_model)
        self.loss_f = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.leaky  = nn.LeakyReLU(negative_slope=negative_slope)
        if sim_func.lower() == 's':
            self.sim_func = SimMatrix()
        if sim_func.lower() == 'd':
            self.sim_func = DistMatrix()
            
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc[0].weight.data)
        
    def forward(self, doc_tids, TFs, DFs, labels=None):
        
        result = self.emb_(doc_tids, TFs, DFs)
        
        co_weights  = self.sim_func( result['K'], result['Q'] )
        #co_weights  = sim_matrix( result['K'], result['Q'] )
        co_weights[result['pad_mask'].logical_not()] = 0. # Set the 3D-pad mask values to
        #co_weights = self.leaky(co_weights)
        
        #weights = torch.relu(co_weights).sum(axis=2) / result['doc_sizes']
        weights = co_weights.sum(axis=2) / result['doc_sizes']
        weights[result['bx_packed']] = float('-inf') # Set the 2D-pad mask values to -inf  (=0 in softmax)
        
        weights = torch.softmax(weights, dim=1)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        weights = weights.view( *weights.shape, 1 )
        
        docs_h = torch.softmax(co_weights, dim=-1) @ result['V']
        docs_h = docs_h * weights
        docs_h = docs_h.sum(axis=1)
        docs_h = self.drop_(docs_h)
        #docs_h = F.dropout( docs_h, p=self.drop_, training=self.training )
        logits = self.fc(docs_h)
        
        
        result_ = {}
        result_['docs_h'] = docs_h
        result_['logits'] = logits
        result_['weights'] = weights
        result_['co_weights'] = co_weights
        
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

class AttentionTFIDFClassifier(BaseEstimator):
    def __init__(self,
                 tokenizer=None, tokenizer_params={ "mindf": 2, "stopwordsSet": None, "model": 'topk', "k": 512, "vocab_max_size": 500000, "ngram_range": (1,2) },
                 transfom_conf={ "gamma": 3., "hiddens": 300, 'att_model': 'AA'}, device='cuda:0',
                 weight_decay=5e-3, lr=5e-3, patience=3, batch_size=16, nepochs=50, max_drop=.75,
                seed=42, verbosity=2):
        self.device = device
        self.transfom_conf = transfom_conf
        self.weight_decay = weight_decay
        self.lr = lr
        self.patience = patience
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.max_drop = max_drop
        self.tknz = tokenizer
        if tokenizer is None:
            self.tknz = Tokenizer(**tokenizer_params)

        self.seed = seed
        self.verbosity = verbosity

        def collate(X):
            doc_tids, TFs, DFs = self.tknz.transform(X, verbose=False)
            doc_tids = pad_sequence(list(map(torch.LongTensor, doc_tids)), batch_first=True, padding_value=0)

            TFs = list(map(lambda x: np.array(np.sqrt(x)), TFs))
            TFs = pad_sequence(list(map(torch.tensor, TFs)), batch_first=True, padding_value=0)
            TFs = torch.LongTensor(TFs.round().long())

            DFs = list(map(lambda x: np.array(np.log2(x+1)), DFs))
            DFs = pad_sequence(list(map(torch.tensor, DFs)), batch_first=True, padding_value=0)
            DFs = torch.LongTensor(DFs.round().long())
                    
            result = { 'doc_tids':  doc_tids, 'TFs': TFs, 'DFs': DFs }
            return result
        def collate_train(params):
            X, y = list(zip(*params))
            result = collate(X)
            result['labels'] = torch.LongTensor( self.tknz.le.transform(y) )
            return result
        self.collate = collate
        self.collate_train = collate_train
        self.model = None
    def train(self, fold):
        seed_everything(self.seed)
        self.transfom_conf['vocab_size'] = self.tknz.vocab_size
        self.transfom_conf['nclass'] = self.tknz.n_class
        self.model = AttentionTFIDF(**self.transfom_conf).to(self.device)
        optimizer = optim.AdamW( self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.95, patience=self.patience)
        best_loss = 99999.
        best_acc  = 0.
        epoch_loss = []
        epoch_acc  = []
        counter = 1
        dl_val = DataLoader(list(zip(fold.X_val, fold.y_val)), batch_size=self.batch_size,
                                shuffle=False, collate_fn=(self.collate_train))
        with tqdm(total=self.nepochs, desc='First batch', position=self.verbosity - 2, disable=(self.verbosity - 2)<0) as spbar, tqdm(total=len(fold.y_train)+len(fold.y_val), position=self.verbosity - 1, disable=(self.verbosity - 1)<0, smoothing=0., desc=f"V-ACC={best_acc:.3} L={best_loss:.6}") as pbar:
            for e in range(self.nepochs):
                pbar.reset()
                dl_train = DataLoader(list(zip(fold.X_train, fold.y_train)), batch_size=self.batch_size,
                                        shuffle=True, collate_fn=self.collate_train)
                loss_train  = 0.
                total = 0.
                correct  = 0.
                self.model.train()
                y_true = []
                y_preds = []
                self.tknz.model = 'sample' #######################################################################################################################################
                for i, data in enumerate(dl_train):
                    data = { k: v.to(self.device) for (k,v) in data.items() }

                    result = self.model( **data )
                    loss   = result['loss']

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_train    += result['loss'].item()

                    y_pred         = result['logits'].argmax(axis=-1)
                    correct       += (y_pred == data['labels']).sum().item()
                    total         += len(data['labels'])

                    y_true.extend(list(data['labels'].cpu()))
                    y_preds.extend(list(y_pred.cpu()))

                    self.model.drop_.p  = (correct/total)*self.max_drop
                    pbar.update( len(data['labels']) )
                    pbar.desc = f"t-ACC: {(correct/total):.3} Drp: {self.model.drop_.p:.3} L={(loss_train/(i+1)):.6} iter={i+1}"

                    #print(, end=f"{ ''.join([' ']*100) }\r")
                    del result, data

                f1_ma = f1_score(y_true, y_preds, average='macro')*100.
                f1_mi = f1_score(y_true, y_preds, average='micro')*100.
                loss_train = loss_train/(i+1)
                total = 0.
                correct  = 0.
                loss_val = 0.
                y_true  = []
                y_preds = [] 
                self.model.eval()
                self.tknz.model = 'topk'
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
                loss_val   = (loss_val/(i+1)) / ( f1_ma + f1_mi )
                epoch_loss.append( loss_val )
                epoch_acc.append( (f1_mi, f1_ma) )
                if best_loss-loss_val > 0.0001 :
                    best_loss = loss_val
                    counter = 1
                    best_acc = correct/total
                    best_model = copy.deepcopy(self.model).to('cpu')
                    spbar.desc = f"*-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) M={loss_val:.5} (BESTACC:{best_acc:.5})"
                elif counter > (3*self.patience):
                    break
                else:
                    spbar.desc = f"v-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) M={loss_val:.5} (BESTACC:{best_acc:.5})"
                    counter += 1
                scheduler.step(loss_val)
                spbar.update(1)
        self.model = copy.deepcopy(best_model).to(self.device)
        result = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc, 'best_loss': best_loss, 'best_acc': best_acc}
        for (k,v) in result.items():
            print(k, v)
        return result
    def save(self, output_path):
        import pickle
        from os import path
        with open(path.join(output_path, 'tknz.pkl'), 'wb') as file_out:
            pickle.dump(self.tknz, file_out)
        torch.save(self.model.state_dict(), path.join(output_path,'model.pkl'))
    def predict(self, X):
        if self.model is None:
            raise Exception("Not trained yet!!")
        
        dl_test = DataLoader(X, batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.collate)
        self.model.eval()
        self.tknz.model = 'topk'
        y_preds = [] 
        for i, data in enumerate(dl_test):
            data = { k: v.to(self.device) for (k,v) in data.items() }
            result = self.model( **data )
            y_pred      = result['logits'].argmax(axis=-1)
            y_preds.extend(list(y_pred.cpu()))

        return np.array(y_preds)