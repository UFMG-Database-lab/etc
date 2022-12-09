
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np
from multiprocessing import cpu_count
from fasttext import train_supervised

def save_file(filename, content):
    with open(filename, 'w') as filout:
        for line in content:
            filout.write(line + '\n')

from unicodedata import normalize as UNI_NORMALIZE
def filter_text(doc):
    doc = UNI_NORMALIZE('NFC', doc)
    doc = doc.lower()
    for c in ["'",'"',".",",",":","!","#","?","$","%","^","&","*","(",")", "\n"]: # Unefficient way V0
        doc = doc.replace(c, ' ')
    return doc
    
TMP_IN_TRAIN_FILE = '/tmp/train_fasttext_in'
class FastTextSKL(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, model='skipgram', lr=0.05, dim=300, ws=5, epoch=5,
                minCount=5, minn=3, maxn=6, neg=5, minCountLabel=0,
                wordNgrams=1, loss='ns', bucket=2000000,
                n_jobs=cpu_count(), lrUpdateRate=100, t=0.0001, verbose=True ):
        self.lrUpdateRate=lrUpdateRate
        self.dim=dim
        self.epoch=epoch
        self.lr=lr
        self.t=t
        self.maxn=maxn
        self.n_jobs=n_jobs
        self.wordNgrams=wordNgrams
        self.ws=ws
        self.minn=minn
        self.verbose=verbose
        self.loss=loss
        self.minCount=minCount
        self.neg=neg
        self.bucket=bucket
        self.minCountLabel = minCountLabel
        self._pretrainedVectors_ = ''

        self.model_ft = None
        self.key = None

        pass
    
    def uniquekey(self):
        from datetime import datetime
        dt = datetime.now()
        return str(dt.microsecond)
    def convert_to_file(self, X, y):
        result = map(filter_text, X)

        if y is not None:
            result = map(' '.join, zip( map(lambda l: "__label__"+str(l), y), result ))

        save_file(TMP_IN_TRAIN_FILE+self.key, result)

        return TMP_IN_TRAIN_FILE+self.key
    def fit(self, X, y):
        self.key = self.uniquekey()
        input_file = self.convert_to_file( X, y )
        
        self.model_ft = train_supervised( input=input_file, 
                    lr=self.lr, dim=self.dim, ws=self.ws, epoch=self.epoch,
                    minCount=self.minCount, minCountLabel=self.minCountLabel,
                    minn=self.minn, maxn=self.maxn, neg=self.neg,
                    wordNgrams=self.wordNgrams, loss=self.loss, bucket=self.bucket,
                    thread=self.n_jobs, lrUpdateRate=self.lrUpdateRate, t=self.t,
                    label="__label__", verbose=self.verbose,
                    pretrainedVectors=self._pretrainedVectors_ )

        return self

    def predict(self, X):
        X = list(map(filter_text, X))
        y_pred = self.model_ft.predict(X)[0]
        y_pred = list(map(lambda y: int(y[0].replace('__label__', '')), y_pred))
        return y_pred