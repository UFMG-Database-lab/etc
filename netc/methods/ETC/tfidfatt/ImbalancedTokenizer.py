from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import torch
from tqdm.auto import tqdm
import numpy as np
import math
from torch.nn.utils.rnn import  pad_sequence
from multiprocessing import cpu_count
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


class ImbalancedTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, imbalancer:str='random', with_CLS=False, **kargs):
        self.with_CLS   = with_CLS
        self.le = LabelEncoder()
        self.kargs = kargs
        self.imbalancer = imbalancer
        self.vectorizer = CountVectorizer(**kargs)
        if imbalancer is None:
            self.oversampler = None
        elif imbalancer.lower() == 'adasyn':
            self.oversampler = ADASYN(random_state=42, n_jobs=cpu_count(), n_neighbors=1, sampling_strategy='minority')
        elif imbalancer.lower() == 'smote':
            self.oversampler = SMOTE(random_state=42, n_jobs=cpu_count())
        elif imbalancer.lower() == 'random':
            self.oversampler = RandomOverSampler(random_state=42)
        else:
            raise NotImplemented(f"Sample not found: {imbalancer}. [None, 'adasyn', 'smote', 'random']")
    def fit(self, X, y):
        X_data = self.vectorizer.fit_transform(X)
        self.V = X_data.shape[1]+1+int(self.with_CLS)    # {<PAD>: 0, <CLS>: 1, ...} 
        y = self.le.fit_transform(y)
        self.L = len(self.le.classes_)
        self._computeDF(X_data)
        if self.oversampler is not None:
            ext_X, self.ext_y = self.oversampler.fit_resample(X_data, y)
            self.ext_X = self._toArray(ext_X)
        else:
            self.ext_X, self.ext_y = self._toArray(X_data), y
        self.N = X_data.shape[0]
        self.ext_N = len(self.ext_X)
        return self
    def _toArray(self, data):
        if self.with_CLS:
            result = [ [(1,len(data[r].data),0)] for r in range(data.shape[0])] # [ [(<CLS>,|d|,1)] ] 
        else:
            result = [ [] for _ in range(data.shape[0])]
        for (tf, did, tid) in zip(data.data, *data.nonzero()):
            result[did].append( (tid+1+int(self.with_CLS), tf, self.DF[tid]) )   # {<CLS>: 1, <PAD>: 0, ...} 
        return result
    def transform(self, X):
        data = self.vectorizer.transform(X)
        return self._toArray(data)
    def _computeDF(self, X):
        X = csr_matrix(X)
        X.data = np.ones_like(X.data)
        self.DF = np.array(X.sum(axis=0))[0,:]
        self.sizes = np.array(X.sum(axis=1))[:,0]
        self.max_size = int(np.percentile(self.sizes, 90))
    def _getFeatures(self, docs):
        docs = map(lambda d: list(sorted(d, key=lambda t: (math.sqrt(t[1])/math.log2(t[2]+2)), reverse=True)), docs)
        docs    = list(map(lambda doc: list(zip(*doc)), docs))
        tids    = [ np.array(item[0])[:self.max_size] if len(item) > 0 else np.array([0]) for item in docs ]
        TF      = [ np.array(item[1])[:self.max_size] if len(item) > 0 else np.array([0]) for item in docs ]
        DF      = [ np.array(item[2])[:self.max_size] if len(item) > 0 else np.array([0]) for item in docs ]
        return tids, TF, DF
    def _toTorch(self, tids, TFs, DFs):
        tids = pad_sequence(list(map(torch.LongTensor, tids)), batch_first=True, padding_value=0)

        TFs = list(map(lambda x: np.array(np.sqrt(x)), TFs))
        TFs = pad_sequence(list(map(torch.tensor, TFs)), batch_first=True, padding_value=0)
        TFs = torch.LongTensor(TFs.round().long())

        DFs = list(map(lambda x: np.array(np.log2(x+1)), DFs))
        DFs = pad_sequence(list(map(torch.tensor, DFs)), batch_first=True, padding_value=0)
        DFs = torch.LongTensor(DFs.round().long())

        result = { 'doc_tids':  tids, 'TFs': TFs, 'DFs': DFs }
        return result
    def collate_train(self, params):
        docs, y = list(zip(*params))
        tids, TF, DF = self._getFeatures(docs)
        result  = self._toTorch(tids, TF, DF)
        result['labels'] = torch.LongTensor( y )
        return result
    def collate_val(self, params):
        docs, y = list(zip(*params))
        result = self.collate(docs)
        result['labels'] = torch.LongTensor( self.le.transform(y) )
        return result
    def collate(self, X):
        docs = self.transform(X)
        tids, TF, DF = self._getFeatures(docs)
        result = self._toTorch(tids, TF, DF)
        return result