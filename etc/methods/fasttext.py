from base import save_file, filter_text, remove_if_exists
import fasttext
from os import path, remove
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class FastTextSupervised(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, pretrainedVectors="", lr=0.1, dim=100, ws=5, epoch=5, minCount=1,
                        minCountLabel=1, minn=0, maxn=0, neg=5,
                        wordNgrams=1, loss='softmax', bucket=2000000, n_jobs=cpu_count(),
                        lrUpdateRate=100, t=0.0001, label='__label__', verbose=0):
        #super().__init__( "FastText_Supervised", use_validation=False )

        self.TMP_IN_TRAIN_FILE = '/tmp/train_fasttext_in_'+str(datetime.now())

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
        self.model_ft = None
        self.label = label
        self.minCountLabel = minCountLabel
        self._pretrainedVectors_ = pretrainedVectors
        self.pretrainedVectors = path.basename(self._pretrainedVectors_)
    
    def convert_to_file(self, X_train, y_train):
        result = X_train

        if y_train is not None:
            _label = self.label
            if self.label is None:
                _label = "__label__"
            result = map(' '.join, zip( map(lambda x: _label+str(x), y_train), result ))

        save_file(self.TMP_IN_TRAIN_FILE, result)

    def __del__(self):
        remove_if_exists(self.TMP_IN_TRAIN_FILE)
        
    def fit(self, X_train, y_train): 
        self.convert_to_file( X_train, y_train )
        
        self.model_ft = fasttext.train_supervised( input=self.TMP_IN_TRAIN_FILE,
                    lr=self.lr, dim=self.dim, ws=self.ws, epoch=self.epoch,
                    minCount=self.minCount, minCountLabel=self.minCountLabel,
                    minn=self.minn, maxn=self.maxn, neg=self.neg,
                    wordNgrams=self.wordNgrams, loss=self.loss, bucket=self.bucket,
                    thread=self.n_jobs, lrUpdateRate=self.lrUpdateRate, t=self.t,
                    label=self.label, verbose=self.verbose,
                    pretrainedVectors=self._pretrainedVectors_ )

        return self

    def predict(self, X):
        if self.model_ft is None:
            raise Exception("Model not fitted yet!")
            
        filtered_X = filter_text(X)
        bla = list(map(self.model_ft.predict, filtered_X))
        y_preds = list(list(zip(*bla))[0])

        replacer_int = lambda x: int(x[0].replace(self.label,''))
        y_preds = list(map(replacer_int, y_preds))
        
        return np.array(y_preds)


    def transform(self, X):
        if self.model_ft is None:
            raise Exception("Model not fitted yet!")

        filtered_X = filter_text(X)
        X_transformed = list(map(self.model_ft.get_sentence_vector, filtered_X))

        return np.array(X_transformed)
