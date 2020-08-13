from hyperparams_search.hyperparam_search import GridSearchHyperparamSearch
from hyperparams_search.hyperparam_search import RandomSearchHyperparamSearch
from hyperparams_search.hyperparam_search import BayesianHyperparamSearch
import numpy as np

default_cls = {
    'fasttext': {
        'classpath': 'methods.fasttext.FastTextSupervised',
        'init_params': { 'loss': 'hs', 'epoch': 1 }, 
        'name_params': [ 'loss', 'epoch', 'lr' ]
    },
    'lstm': {
        'classpath': 'methods.lstm.LSTMClassifier',
        'init_params': { 'epochs': 1 }, 
        'name_params': [ 'patience', 'epochs', 'batch_size' ]
    },
    'swem': {
        'classpath': 'methods.swem.SWEM',
        'init_params': { 'max_epochs': 1 }, 
        'name_params': [ 'max_epochs', 'pacience' ]
    },
    'svm': {
        'classpath': 'sklearn.svm.SVC',
        'init_params': {'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
			 'degree': 3, 'shrinking': True, 
			 'decision_function_shape': None, 'random_state': None, 
			 'tol': 0.001, 'cache_size': 25000, 'coef0': 0.0, 'gamma': 'auto', 
			 'class_weight': None,'random_state': 42}, 
        'name_params': [ 'C' ]
    }
}
default_repr = {
    'tfidf': {
        'classpath': 'methods.frequency_based.TFIDFRepresentation',
        'init_params': { "min_df": 2, "stopwords": "english" }, 
        'name_params': [ "min_df","stopwords" ]
    },
    'fasttext': {
        'classpath': 'methods.fasttext.FastTextSupervised',
        'init_params': { 'loss': 'hs', 'epoch': 1 }, 
        'name_params': [ 'loss', 'epoch', 'lr' ]
    },
    'swem': {
        'classpath': 'methods.swem.SWEM',
        'init_params': { 'max_epochs': 1 }, 
        'name_params': [ 'max_epochs', 'pacience' ]
    }
}
default_hyperparams = {
    #'gs-svm': GridSearchHyperparamSearch( parameters=[{'C': 2.0 ** np.arange(-5, 15, 2)}], scoring='f1_micro', cv=5 ),
    'gs-svm': GridSearchHyperparamSearch( parameters=[{'C': 2.0 ** np.arange(-5, 2, 2)}], scoring='f1_micro', cv=5 ),
    #'rs-svm': RandomSearchHyperparamSearch( parameters={'C': 2.0 ** np.arange(-5, 15, 2)}, scoring='f1_micro', cv=5 ),
    'rs-svm': RandomSearchHyperparamSearch( parameters={'C': 2.0 ** np.arange(-5, 2, 2)}, n_iter=2, scoring='f1_micro', cv=5 ),
    'bs-svm': BayesianHyperparamSearch( parameters=[{'C': 2.0 ** np.arange(-5, 15, 2)}], scoring='f1_micro', cv=5 )
}