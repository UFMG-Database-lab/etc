import numpy as np
from copy import deepcopy
from nltk.corpus import stopwords as stopwords_by_lang
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
stopwordsSet  = list(set(stopwords_by_lang.words('english')))
stopwordsSet += list(set(stop_words))

svm_desc = {
    'classpath': 'sklearn.svm.SVC',
    'init_params': {'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
        'degree': 3, 'shrinking': True, 
        'decision_function_shape': 'ovo', 'random_state': None, 
        'tol': 0.001, 'cache_size': 25000, 'coef0': 0.0, 'gamma': 'auto', 
        'class_weight': None,'random_state': 42}
}

tfidf_desc = {
    'classpath': 'sklearn.feature_extraction.text.TfidfVectorizer',
    'init_params': { "min_df": 2, "stop_words": "english" }
}

ftfidf_desc = {
    'classpath': 'netc.methods.frequency_based.TFIDFRepresentation',
    'init_params': { "min_df": 2, "stop_words": "english" }
}

tfidfsvm = {
    'classpath': 'netc.trainers.skl_trainers.SklTrainer',
    'init_params': { 'tname': 'TFIDF+SVM',
                     'descriptor': { 'type': 'skl-pipe',
                                     'pipeline': [ tfidf_desc, svm_desc ]
                                    }
                    }
}

bert_tiny = {
    'classpath': 'netc.trainers.hug_trainers.HuggTrainer',
    'init_params': { 'tname': 'BERT-tiny',
                     'descriptor': { 'model_name': 'prajjwal1/bert-tiny',
                                     'training_args': {
                                        'evaluation_strategy':'epoch',
                                        'save_strategy' : "epoch",
                                        'eval_steps': 1,
                                        'weight_decay': 1e-2,
                                        #'dropout': 0.1,
                                        'learning_rate': 5e-5,
                                        'load_best_model_at_end': True,
                                        'per_device_train_batch_size': 32,
                                        'per_device_eval_batch_size': 32,
                                        'num_train_epochs': 16
                                      },
                                      #'callbacks':[ { 'classpath': 'pytorch_lightning.callbacks.early_stopping.EarlyStopping',
                                      #                'init_params': { 'monitor':"val_loss", 'mode':"min", 'patience': 5 }
                                      #               } ]
                                    }
                    }
}

bert = {
    'classpath': 'netc.trainers.hug_trainers.HuggTrainer',
    'init_params': { 'tname': 'Hugg-BERT',
                     'descriptor': { 'model_name': 'bert-base-uncased',
                                     'training_args': {
                                        'evaluation_strategy':'epoch',
                                        'save_strategy' : "epoch",
                                        'eval_steps': 1,
                                        'weight_decay': 1e-2,
                                        'dropout': 0.1,
                                        'learning_rate': 5e-5,
                                        'load_best_model_at_end': True,
                                        'per_device_train_batch_size': 32,
                                        'per_device_eval_batch_size': 32,
                                        'num_train_epochs': 16
                                      },
                                      'callbacks':[ { 'classpath': 'pytorch_lightning.callbacks.early_stopping.EarlyStopping',
                                                      'init_params': { 'monitor':"val_loss", 'mode':"min", 'patience': 5 }
                                                     } ]
                                    }
                    }
}
gridtfidfsvm = {
    'classpath': 'netc.trainers.skl_trainers.CVSklTrainer',
    'init_params': {
        'tname': 'gridTFIDF+SVM',
        'descriptor': { 'skmodel': { 
                                    'type': 'skl-pipe',
                                    'pipeline': [ tfidf_desc, svm_desc ]
                        },
                        'hyperparams': {
                            'tfidfvectorizer__min_df': [1,2,4],
                            'tfidfvectorizer__ngram_range': [(1,1),(1,2)],
                            'svc__C': 2.0 ** np.arange(-5, 10, 2),
                        },
                        'gridparams': {
                            'cv': 5,
                            'n_jobs': -1,
                            'verbose': 4
                        }
                    },
        },
}

bert_skl = {
    'classpath': 'bert_sklearn.BertClassifier',
    'init_params': {'max_seq_length': 256, 'train_batch_size':16, 'epochs':5}
}

bert_skl_desc = {
    'classpath': 'netc.trainers.skl_trainers.SklTrainer',
    'init_params': { 'tname': 'BERT-SKL',
                     'descriptor': bert_skl
                    }
}

etc_desc = {
    'classpath': 'netc.trainers.ETC_trainer.TrainerETC',
    'init_params': { 'tname': 'EnsembleTC',
                     'descriptor': {
                        'classpath': 'netc.methods.ETCClassifier.ETCClassifier',
                        'init_params': {
                            'tknz': { 
                                    'mindf': 2,
                                    'stopwordsSet': 'both',
                                    'model': 'topk',
                                    'oversample': False, 
                                    'vocab_max_size': 750000,
                                    'ngram_range': (1,2),
                                    'verbose': True
                                },
                            'model': { 
                                "gamma": 5.,
                                "hiddens": 300,
                                'nheads': 12,
                                'att_model': 'AA',
                                'norep': 1
                            },
                            'device': 'cuda',
                            'batch_size': 32,
                            'nepochs': 50,
                            'lr': 5e-3,
                            'weight_decay': 5e-3,
                            'max_drop': .75
                         }
                      }
                    }

}

tfidfatt = {
    'classpath': 'netc.trainers.ETC_trainer.TrainerETC',
    'init_params': { 'tname': 'ETC-TFIDF',
                     'descriptor': {
                        'classpath': 'netc.methods.ETC.tfidfatt.AttTFIDFClassifier.AttTFIDFClassifier',
                        'init_params': {
                            'tknz': { 
                                    'mindf': 2,
                                    'stopwordsSet': 'both',
                                    'model': 'topk',
                                    'oversample': False, 
                                    'vocab_max_size': 1000000,
                                    'ngram_range': (1,2),
                                    'verbose': True
                                },
                            'model': { 
                                "gamma": 5.,
                                'nheads': 16,
                                "hiddens": 512,
                                "norep": 2,
                                'sim_func': 'dist',
                                'att_model': 'AA'
                            },
                            'device': 'cuda',
                            'batch_size': 16,
                            'nepochs': 100,
                            'lr': 5e-3,
                            'weight_decay': 5e-3,
                            'max_drop': .75
                         }
                      }
                    }

}

etc_imb = {
    'classpath': 'netc.trainers.ETC_trainer.TrainerETC',
    'init_params': { 'tname': 'ETC-Imb',
                     'descriptor': {
                        'classpath': 'netc.methods.ETC.tfidfatt.ETCClassifier.ETCClassifier',
                        'init_params': {
                            'tknz': { 
                                'min_df': 2,
                                'max_features': 500_000,
                                'stop_words': stopwordsSet,
                                'ngram_range': (1,2),
                                'with_CLS': False,
                                'imbalancer': None
                            },
                            'model': { 
                                "gamma": 5.,
                                "hiddens": 300,
                                'nheads': 12,
                                'att_model': 'aa',
                                'sim_func': 'dist',
                                'norep': 2
                            },
                            'device': 'cuda',
                            'batch_size': 8,
                            'nepochs': 50, 
                            'lr': 5e-3,
                            'weight_decay': 5e-3,
                            'max_drop': .75
                         }
                      }
                    }

}

drst_etc = deepcopy(etc_imb)
drst_etc['init_params']['tname'] = 'ETC-drastic'
drst_etc['init_params']['descriptor']['init_params']['model']['nheads'] = 100


ce_etc = deepcopy(etc_imb)
ce_etc['init_params']['tname'] = 'ETC-CE'
ce_etc['init_params']['descriptor']['init_params']['model']['gamma'] = 0.

stp_etc = deepcopy(etc_imb)
stp_etc['init_params']['tname'] = 'ETC-not-stpw'
stp_etc['init_params']['descriptor']['init_params']['tknz']['stop_words'] = None

b32_etc = deepcopy(etc_imb)
b32_etc['init_params']['tname'] = 'ETC-b32'
b32_etc['init_params']['descriptor']['init_params']['batch_size'] = 32

b64_etc = deepcopy(etc_imb)
b64_etc['init_params']['tname'] = 'ETC-b64'
b64_etc['init_params']['descriptor']['init_params']['batch_size'] = 64

setc_imb = deepcopy(etc_imb)
setc_imb['init_params']['tname'] = 'sim-ETC'
setc_imb['init_params']['descriptor']['init_params']['model']['sim_func'] = 'sim'

detc_imb = deepcopy(etc_imb)
detc_imb['init_params']['tname'] = 'drop-ETC'
detc_imb['init_params']['descriptor']['init_params']['max_drop'] = .25

self_etc = deepcopy(etc_imb)
self_etc['init_params']['tname'] = 'SelfAtt-ETC'
self_etc['init_params']['descriptor']['init_params']['model']['att_model'] = 'sa'

cross_etc = deepcopy(etc_imb)
cross_etc['init_params']['tname'] = 'CrossAtt-ETC'
cross_etc['init_params']['descriptor']['init_params']['model']['att_model'] = 'ca'

doc_etc = deepcopy(etc_imb)
doc_etc['init_params']['tname'] = 'ETC-WithDocRepr'
doc_etc['init_params']['descriptor']['init_params']['tknz']['with_CLS'] = True

etc_imb_smote = deepcopy(etc_imb)
etc_imb_smote['init_params']['tname'] = 'ETC-SMOTE'
etc_imb_smote['init_params']['descriptor']['init_params']['tknz']['imbalancer'] = 'smote'
#etc_imb_smote['init_params']['descriptor']['init_params']['batch_size'] = 16

etc_imb_ada = deepcopy(etc_imb)
etc_imb_ada['init_params']['tname'] = 'ETC-ADASYN'
#etc_imb_ada['init_params']['descriptor']['init_params']['batch_size'] = 16
etc_imb_ada['init_params']['descriptor']['init_params']['tknz']['imbalancer'] = 'adasyn'

etc_imb_rand = deepcopy(etc_imb)
etc_imb_rand['init_params']['tname'] = 'ETC-Random'
#etc_imb_rand['init_params']['descriptor']['init_params']['batch_size'] = 16
etc_imb_rand['init_params']['descriptor']['init_params']['tknz']['imbalancer'] = 'random'

pte = {
    'classpath': 'netc.trainers.skl_trainers.CVSklTrainer',
    'init_params': {
        'tname': 'PTE-LR',
        'descriptor': { 'skmodel': { 
                                    'type': 'skl-pipe',
                                    'pipeline': [
                                        {'classpath': 'netc.methods.PTE.PTEWrapper.PTEVectorizer', 'init_params': {  } }, 
                                        {'classpath': 'sklearn.linear_model.LogisticRegression',   'init_params': {  } }
                                    ]
                        },
                        'hyperparams': {
                            'logisticregression__C': np.logspace(-4, 4, 50),
                            'logisticregression__penalty': ['none', 'l2']
                        },
                        'gridparams': {
                            'cv': 3,
                            'n_jobs': -1,
                            'verbose': 4
                        }
                    },
        },

}
fasttext = {
    'classpath': 'netc.trainers.skl_trainers.SklTrainer',
    'init_params': { 'tname': 'FastText_Sup',
                     'descriptor': {'classpath': 'netc.methods.fasttext.FastTextSKL',
                                    'init_params': { } }
                    }
    
}

transf_hugging = {
    'classpath': 'netc.trainers.hug_trainers.HuggTrainer',
    'init_params': { 'tname': 'HF-',
                     'descriptor': { 
                         'model_name': None,
                         'batch_size': 32,
                         'max_length': 256,
                         'epochs': 5,
                         'learning_rate': 5e-5
                     }
                    }

}



mbert_hugg = deepcopy(transf_hugging)
mbert_hugg['init_params']['tname'] += 'bert-mini'
mbert_hugg['init_params']['descriptor']['model_name'] = 'prajjwal1/bert-mini'
mbert_hugg['init_params']['descriptor']['epochs'] = 1

bert_hugg = deepcopy(transf_hugging)
bert_hugg['init_params']['tname'] += 'bert'
bert_hugg['init_params']['descriptor']['model_name'] = 'bert-base-uncased'

albert_hugg = deepcopy(transf_hugging)
albert_hugg['init_params']['tname'] += 'albert'
albert_hugg['init_params']['descriptor']['model_name'] = 'albert-base-v2'

roberta_hugg = deepcopy(transf_hugging)
roberta_hugg['init_params']['tname'] += 'roberta'
roberta_hugg['init_params']['descriptor']['model_name'] = 'roberta-base'

xlnet_hugg = deepcopy(transf_hugging)
xlnet_hugg['init_params']['tname'] += 'xlnet'
xlnet_hugg['init_params']['descriptor']['model_name'] = 'xlnet-base-cased'

distilbert_hugg = deepcopy(transf_hugging)
distilbert_hugg['init_params']['tname'] += 'distilbert'
distilbert_hugg['init_params']['descriptor']['model_name'] = 'distilbert-base-uncased'



"""transf_hugg = {
    'classpath': 'netc.trainers.skl_trainers.SklTrainer',
    'init_params': { 'tname': 'hugging',
                     'descriptor': { 
                        'classpath': 'netc.methods.TransformBased.DeepClassifier.DeepClassifier',
                        'init_params': {'batch_num': 16, 'max_len':192, 'epochs':5}
                     }
                    }
}
bert_hugg = deepcopy(transf_hugg)
bert_hugg['init_params']['tname'] = 'bert-hugg-base'
bert_hugg['init_params']['descriptor']['init_params']["deepmethod"] = 'bert'

albert_hugg = deepcopy(transf_hugg)
albert_hugg['init_params']['tname'] = 'albert-hugg-base'
albert_hugg['init_params']['descriptor']['init_params']["deepmethod"] = 'albert'

roberta_hugg = deepcopy(transf_hugg)
roberta_hugg['init_params']['tname'] = 'roberta-hugg-base'
roberta_hugg['init_params']['descriptor']['init_params']["deepmethod"] = 'roberta'

xlnet_hugg = deepcopy(transf_hugg)
xlnet_hugg['init_params']['tname'] = 'xlnet-hugg-base'
xlnet_hugg['init_params']['descriptor']['init_params']["deepmethod"] = 'xlnet'

distilbert_hugg = deepcopy(transf_hugg)
distilbert_hugg['init_params']['tname'] = 'distilbert-hugg-base'
distilbert_hugg['init_params']['descriptor']['init_params']["deepmethod"] = 'distilbert'"""

DESC_CLS = {
    'gridtfidfsvm': gridtfidfsvm, 
    'tfidfsvm': tfidfsvm, 
    'bert': bert,
    'pte': pte,
    'fasttext': fasttext,
    'bert-tiny': bert_tiny,
    'bert-skl': bert_skl_desc,

    'etc-imb': etc_imb,
    'etc-imb-ada': etc_imb_ada,
    'etc-imb-rand': etc_imb_rand,
    'etc-imb-smote': etc_imb_smote,

    'setc': setc_imb,
    'detc': detc_imb,
    'cetc': cross_etc,
    'self_etc': self_etc,
    'doc_etc': doc_etc,
    'b32_etc': b32_etc,
    'b64_etc': b64_etc,
    'stpw_etc': stp_etc,
    'drst_etc': drst_etc,
    'ce_etc': ce_etc,

    "hmbert": mbert_hugg,
    "hbert": bert_hugg,
    "halbert": albert_hugg,
    "hroberta": roberta_hugg,
    "hxlnet": xlnet_hugg,
    "hdistilbert": distilbert_hugg,
}
