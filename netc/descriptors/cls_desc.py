

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
import numpy as np
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
                            'cv': 3,
                            'n_jobs': -1,
                            'verbose': 4
                        }
                    },
        },
}

gridftfidfsvm = {
    'classpath': 'netc.trainers.skl_trainers.CVSklTrainer',
    'init_params': {
        'tname': 'grid-FTFIDF+SVM',
        'descriptor': { 'skmodel': { 
                                    'type': 'skl-pipe',
                                    'pipeline': [ ftfidf_desc, svm_desc ]
                        },
                        'hyperparams': {
                            'tfidfrepresentation__min_df': [1,2,4],
                            'tfidfrepresentation__ngram_range': [(1,1),(1,2)],
                            'svc__C': 2.0 ** np.arange(-5, 10, 2),
                        },
                        'gridparams': {
                            'cv': 3,
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
                                    'vocab_max_size': 500000,
                                    'ngram_range': (1,2),
                                    'verbose': True
                                },
                            'model': { 
                                "gamma": 3.,
                                "hiddens": 300,
                                'att_model': 'AA'
                            },
                            'device': 'cuda',
                            'batch_size': 16,
                            'nepochs': 50,
                            'lr': 5e-3,
                            'weight_decay': 5e-3,
                            'max_drop': .75
                         }
                      }
                    }

}

tfidfatt = {
    'classpath': 'netc.methods.tfidfatt.TrainerAttentionTFIDF.TrainerAttentionTFIDF',
    'init_params': { 'tname': 'NAttTFIDF',
                     'descriptor': {
                        'classpath': 'netc.methods.tfidfatt.AttentionTFIDF.AttTFIDFClassifier',
                        'init_params': {
                            'tknz': { 
                                    'mindf': 2,
                                    'stopwordsSet': 'both',
                                    'model': 'topk',
                                    'oversample': False, 
                                    'vocab_max_size': 500000,
                                    'ngram_range': (1,2),
                                    'verbose': True
                                },
                            'model': { 
                                "gamma": 3.,
                                "hiddens": 300,
                                'att_model': 'AA'
                            },
                            'device': 'cuda',
                            'batch_size': 16,
                            'nepochs': 50,
                            'lr': 5e-3,
                            'weight_decay': 5e-3,
                            'max_drop': .75
                         }
                      }
                    }

}

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

transf_hugg = {
    'classpath': 'netc.trainers.skl_trainers.SklTrainer',
    'init_params': { 'tname': 'hugging',
                     'descriptor': { 
                        'classpath': 'netc.methods.TransformBased.DeepClassifier.DeepClassifier',
                        'init_params': {'batch_num': 16, 'max_len':192, 'epochs':5}
                     }
                    }
}
from copy import deepcopy
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
distilbert_hugg['init_params']['descriptor']['init_params']["deepmethod"] = 'distilbert'

DESC_CLS = {
    'gridftfidfsvm': gridftfidfsvm,
    'gridtfidfsvm': gridtfidfsvm, 
    'tfidfsvm': tfidfsvm, 
    'bert': bert,
    'pte': pte,
    'etc': etc_desc,
    'fasttext': fasttext,
    'bert-tiny': bert_tiny,
    'bert-skl': bert_skl_desc,
    'tfidfatt': tfidfatt,

    "hbert": bert_hugg,
    "halbert": albert_hugg,
    "hroberta": roberta_hugg,
    "hxlnet": xlnet_hugg,
    "hdistilbert": distilbert_hugg,
}