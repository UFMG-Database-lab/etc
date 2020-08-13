# -*- coding: utf-8 -*-
from tqdm import tqdm
import importlib
from datetime import datetime
from time import time
from os import path
import numpy as np
import traceback
from base import create_path, save_json, load_json, Representation
from base import Dataset, dump_svmlight_file_gz, is_jsonable

def get_name(learnable, name_params, name_learnable=None):
    name = name_learnable
    if name is None:
        name = learnable.name_learnable
    if name_params is not None:
        for param in name_params:
            name += f"_{param}={learnable.get_params()[param]}"
    return name

class ExecutorE2E(object):
    def __init__(self, dataset, output_path, method_descriptor, name_fold,
                    hyperparam_search=None, save_model=False, predict_proba=False, force=None, silence=False):
        self.type_executor = 'e2e'

        if isinstance(dataset, str) and path.exists(dataset):
            self.dataset = Dataset(dataset)
        else:
            self.dataset = dataset
        
        self.save_model = save_model
        self.predict_proba = predict_proba
        self.name_fold = name_fold
        self.nfolds = self.dataset.nfold(self.name_fold)
        self.hyperparam_search = hyperparam_search
        self.silence = silence

        self.force = force

        self.init_params =  {}
        if 'init_params' in method_descriptor:
            self.init_params = method_descriptor['init_params']

        self.classpath = method_descriptor['classpath']
        self.module_name, self.class_name = self.classpath.rsplit('.', 1)
        e2e_module = importlib.import_module(self.module_name)
        self._e2e_class_ = e2e_module.__getattribute__(self.class_name)
        self.e2e = self._e2e_class_(**self.init_params)
        self.all_params = self.e2e.get_params()

        name_params = None if 'name_params' not in method_descriptor else method_descriptor['name_params']
        name_alias  = self.class_name if 'alias' not in method_descriptor else method_descriptor['alias']
        self.name_method = get_name(self.e2e, name_params, name_learnable=name_alias)
        
        self.output_path = path.join( output_path, self.dataset.dname,
                            f'fold-{self.name_fold}', self.name_method)
        create_path(self.output_path)

        self.load_or_create_configuration()
    
    def __str__(self):
        return f'<E2E ({self.name_method})>'
    
    def load_or_create_configuration(self):
        config_file = path.join(self.output_path, 'result.json')
        if path.exists(config_file):
            self.config = load_json(config_file)
            ## TODO: asserts
        else:
            self.config = dict()
            self.config['input_dataset'] = self.dataset.dataset_path
            self.config['dname'] = self.dataset.dname
            self.config['nfolds'] = self.nfolds
            self.config['splits'] = self.dataset[self.name_fold]
            self.config['name_fold'] = self.name_fold
            self.config['type_executor'] = self.type_executor
            self.config['e2e'] = {} 

            self.config['e2e']['classpath'] = self.classpath
            self.config['e2e']['name_e2e'] = self.e2e.name_e2e
            self.config['e2e']['with_val'] = self.e2e.use_validation
            self.config['e2e']['name_method'] = self.name_method
            self.config['e2e']['save_model'] = self.save_model
            self.config['e2e']['creation_date'] = str(datetime.now())
            self.config['e2e']['general_params'] = { k: v if is_jsonable(v) else str(v) for (k,v) in self.all_params.items() } 
            self.config['e2e']['init_params'] = { k: v if is_jsonable(v) else str(v) for (k,v) in self.init_params.items() }

            self.config['e2e']['time_init'] = [None] * self.nfolds
            self.config['e2e']['time_fit'] = [None] * self.nfolds
            self.config['e2e']['time_train'] = [None] * self.nfolds
            self.config['e2e']['time_pred_test'] = [None] * self.nfolds
            self.config['e2e']['y_pred_test'] = [None] * self.nfolds
            self.config['e2e']['y_true_test'] = [None] * self.nfolds

            self.config['e2e']['status'] = ["TODO"] * self.nfolds

            if self.e2e.use_validation:
                self.config['e2e']['y_pred_val'] = [None] * self.nfolds
                self.config['e2e']['y_true_val'] = [None] * self.nfolds
            if self.predict_proba:
                self.config['e2e']['y_pred_proba_test'] = [None] * self.nfolds
            if self.hyperparam_search is not None:
                self.config['e2e']['hyperparam_search'] = self.hyperparam_search.config
                self.config['e2e']['time_hyperparam_search'] = [None] * self.nfolds
                self.config['e2e']['best_params'] = [None] * self.nfolds
                self.config['e2e']['all_searched_params'] = [None] * self.nfolds
            self.save_config()

    def run(self):
        if self.force is not None:
            if self.force == 'all':
                force_run = [ f for f in range(self.nfolds) ]
            else:
                force_run = self.force
        else:
            force_run = [ f for f in range(self.nfolds) if self.config['e2e']['status'][f] != "DONE" ]
        
        for f in tqdm(force_run, desc=f"Running on folds ({self.dataset.dname})", disable=self.silence, position=2):
            fold = self.dataset.get_fold(f, self.name_fold, with_val=self.e2e.use_validation)
            try:
                del self.e2e

                t_train = time()
                t_measure = time()
                self.e2e = self._e2e_class_(**self.init_params)
                self.config['e2e']['time_init'][f] = time() - t_measure

                if self.hyperparam_search is not None:
                    # TODO: implement the abstract version of hyperparam_search
                    t_measure = time()
                    result = self.hyperparam_search.run(self.e2e, fold)
                    self.config['e2e']['time_hyperparam_search'][f] = time() - t_measure

                    self.e2e.set_params( **result.best_params )
                    self.config['e2e']['best_params'][f] = self.e2e.get_params()
                    self.config['e2e']['all_searched_params'][f] = result.all_results

                t_measure = time()
                self.e2e.fit(fold)
                self.config['e2e']['time_fit'][f] = time() - t_measure
                self.config['e2e']['time_train'][f] = time() - t_train

                t_measure = time()
                y_pred = self.e2e.predict( fold.X_test )
                self.config['e2e']['time_pred_test'][f] = time() - t_measure

                self.config['e2e']['y_pred_test'][f] = y_pred
                self.config['e2e']['y_true_test'][f] = fold.y_test

                if self.e2e.use_validation:
                    self.config['e2e']['y_pred_val'][f] = self.e2e.predict( fold.X_val )
                    self.config['e2e']['y_true_val'][f] = fold.y_val

                if self.predict_proba:
                    t_measure = time()
                    y_pred_proba = self.e2e.predict_proba( fold.X_test )
                    self.config['e2e']['time_pred_test'][f] = time() - t_measure
                    self.config['e2e']['y_pred_proba_test'] = y_pred_proba
                
                self.config['e2e']['status'][f] = "DONE"
                self.save_config()
            except KeyboardInterrupt:
                self.save_config()
                break
            except Exception as e:
                self.config['e2e']['status'][f] = traceback.format_exc()
                continue
        del self.e2e

    def save_config(self):
        save_json(path.join(self.output_path, 'result.json'), self.config)
class ExecutorRepr(object):
    def __init__(self, dataset, output_path, method_descriptor, name_fold, save_model=False, force=None, silence=False):
        if isinstance(dataset, str) and path.exists(dataset):
            self.dataset = Dataset(dataset)
        else:
            self.dataset = dataset
        
        self.save_model = save_model
        self.name_fold = name_fold
        self.nfolds = self.dataset.nfold(self.name_fold)
        self.silence = silence

        self.force = force

        self.init_params =  {}
        if 'init_params' in method_descriptor:
            self.init_params = method_descriptor['init_params']

        self.classpath = method_descriptor['classpath']
        self.module_name, self.class_name = self.classpath.rsplit('.', 1)
        Representation_module = importlib.import_module(self.module_name)
        self._Representation_class_ = Representation_module.__getattribute__(self.class_name)
        self.representation = self._Representation_class_(**self.init_params)
        self.all_params = self.representation.get_params()

        name_params = None if 'name_params' not in method_descriptor else method_descriptor['name_params']
        name_alias  = self.class_name if 'alias' not in method_descriptor else method_descriptor['alias']
        self.name_method = get_name(self.representation, name_params, name_learnable=name_alias)
        self.use_validation = False if 'use_val' not in method_descriptor else method_descriptor['use_val']

        self.output_path = path.join( output_path, self.dataset.dname,
                            f'fold-{self.name_fold}', self.name_method)
        create_path(self.output_path)

        self.load_or_create_configuration()
    
    def __str__(self):
        return f'<Representation ({self.name_method})>'
    
    def load_or_create_configuration(self):
        config_file = path.join(self.output_path, 'result.json')
        if path.exists(config_file):
            self.config = load_json(config_file)
        else:
            self.config = dict()
            self.config['input_dataset'] = self.dataset.dataset_path
            self.config['dname'] = self.dataset.dname
            self.config['nfolds'] = self.nfolds
            self.config['splits'] = self.dataset[self.name_fold]
            self.config['name_fold'] = self.name_fold

            self.config['representation'] = {}

            self.config['representation']['name_method'] = self.name_method
            self.config['representation']['classpath'] = self.classpath
            self.config['representation']['name_representation'] = self.name_alias
            self.config['representation']['with_val'] = self.use_validation
            self.config['representation']['save_model'] = self.save_model
            self.config['representation']['creation_date'] = str(datetime.now())
            self.config['representation']['general_params'] = { k: v if is_jsonable(v) else str(v) for (k,v) in self.all_params.items() } 
            self.config['representation']['init_params'] = { k: v if is_jsonable(v) else str(v) for (k,v) in self.init_params.items() }

            self.config['representation']['time_init'] = [None] * self.nfolds
            self.config['representation']['time_fit'] = [None] * self.nfolds
            self.config['representation']['time_train'] = [None] * self.nfolds
            self.config['representation']['time_transform_train'] = [None] * self.nfolds
            self.config['representation']['time_transform_test'] = [None] * self.nfolds

            if self.use_validation:
                self.config['representation']['time_transform_val'] = [None] * self.nfolds
            
            self.config['representation']['status'] = ["TODO"] * self.nfolds
            self.save_config()

    def run(self):
        if self.force is not None:
            if self.force == 'all':
                force_run = [ f for f in range(self.nfolds) ]
            else:
                force_run = self.force
        else:
            force_run = [ f for f in range(self.nfolds) if self.config['representation']['status'][f] != "DONE" ]
        
        for f in tqdm(force_run, desc=f"Running on folds ({self.dataset.dname})", disable=self.silence, position=2):
            fold = self.dataset.get_fold(f, self.name_fold, with_val=self.use_validation)
            
            #try:
            del self.representation

            t_train = time()
            t_measure = time()
            self.representation = self._Representation_class_(**self.init_params)
            self.config['representation']['time_init'][f] = time() - t_measure

            if self.use_validation:
                t_measure = time()
                self.representation.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)
                self.config['representation']['time_fit'][f] = time() - t_measure
                self.config['representation']['time_train'][f] = time() - t_train
            
                t_measure = time()
                X_val_vectors = self.representation.transform( fold.X_val )
                self.config['representation']['time_transform_val'][f] = time() - t_measure

                # TODO: Ideally it is interesting to reconstruct the training
                # in the same order as the original training set.
                X_train_vectors = np.concatenate((X_train_vectors, X_val_vectors))
                y_train = np.concatenate((fold.y_train, fold.y_val))
            else:
                t_measure = time()
                self.representation.fit(fold.X_train, fold.y_train)
                self.config['representation']['time_fit'][f] = time() - t_measure
                self.config['representation']['time_train'][f] = time() - t_train

                t_measure = time()
                X_train_vectors = self.representation.transform( fold.X_train )
                self.config['representation']['time_transform_train'][f] = time() - t_measure
                y_train = fold.y_train

            self.save_svmlight(X_train_vectors, y_train, f'train_{f}.gz')

            t_measure = time()
            X_test_vectors = self.representation.transform( fold.X_test )
            self.config['representation']['time_transform_test'][f] = time() - t_measure
            self.save_svmlight(X_test_vectors, fold.y_test, f'test_{f}.gz')
            
            self.config['representation']['status'][f] = "DONE"
            self.save_config()
            #except KeyboardInterrupt:
            #    self.save_config()
            #    break
            #except Exception as e:
            #    self.config['status'][f] = traceback.format_exc()
            #    continue
        del self.representation

    def save_svmlight(self, X, y, namefile):
        dump_svmlight_file_gz(X, y, path.join(self.output_path, namefile))

    def save_config(self):
        save_json(path.join(self.output_path, 'result.json'), self.config)
class ExecutorClassifier(object):
    def __init__(self, representation, output_path, method_descriptor, 
                        hyperparam_search=None, save_model=False, predict_proba=False, force=None, silence=False):
        self.type_executor = '2p'

        if isinstance(representation, str) and path.exists(representation):
            self.representation = Representation(representation)
        else:
            self.representation = representation
        
        self.save_model = save_model
        self.predict_proba = predict_proba
        self.nfolds = self.representation.nfold
        self.hyperparam_search = hyperparam_search
        self.silence = silence

        self.force = force

        self.init_params =  {}
        if 'init_params' in method_descriptor:
            self.init_params = method_descriptor['init_params']

        self.classpath = method_descriptor['classpath']
        self.module_name, self.class_name = self.classpath.rsplit('.', 1)
        Classifier_module = importlib.import_module(self.module_name)
        self._Classifier_class_ = Classifier_module.__getattribute__(self.class_name)
        self.classifier = self._Classifier_class_(**self.init_params)
        self.all_params = self.classifier.get_params()

        name_params = None if 'name_params' not in method_descriptor else method_descriptor['name_params']
        name_alias  = self.class_name if 'alias' not in method_descriptor else method_descriptor['alias']
        self.name_method = get_name(self.classifier, name_params, name_learnable=name_alias)

        self.output_path = path.join( output_path, self.representation.dname,
                            f'fold-{self.representation.name_fold}', self.representation.name_repr, self.name_method)
        create_path(self.output_path)

        self.load_or_create_configuration()
    
    def __str__(self):
        return f'<Classifier ({self.name_method})>'
    
    def load_or_create_configuration(self):
        self.config = self.representation.config

        config_file = path.join(self.output_path, 'result.json')
        if path.exists(config_file):
            self.config = load_json(config_file)
        else:
            self.config['type_executor'] = self.type_executor
            self.config['classification'] = {}

            self.config['classification']['classpath'] = self.classpath
            self.config['classification']['name_classifier'] = self.class_name
            self.config['classification']['name_method'] = self.name_method
            self.config['classification']['save_model'] = self.save_model
            self.config['classification']['creation_date'] = str(datetime.now())
            self.config['classification']['general_params'] = { k: v if is_jsonable(v) else str(v) for (k,v) in self.all_params.items() } 
            self.config['classification']['init_params'] = { k: v if is_jsonable(v) else str(v) for (k,v) in self.init_params.items() }

            self.config['classification']['time_init'] = [None] * self.nfolds
            self.config['classification']['time_fit'] = [None] * self.nfolds
            self.config['classification']['time_train'] = [None] * self.nfolds
            self.config['classification']['time_pred_test'] = [None] * self.nfolds
            self.config['classification']['y_pred_test'] = [None] * self.nfolds
            self.config['classification']['y_true_test'] = [None] * self.nfolds

            self.config['classification']['status'] = ["TODO"] * self.nfolds

            if self.hyperparam_search is not None:
                self.config['classification']['hyperparam_search'] = self.hyperparam_search.name_method
                self.config['classification']['hyperparam_search_params'] = self.hyperparam_search.get_params()
                self.config['classification']['time_hyperparam_search'] = [None] * self.nfolds
                self.config['classification']['best_params'] = [None] * self.nfolds

            if self.predict_proba:
                self.config['classification']['y_pred_proba_test'] = [None] * self.nfolds
            self.save_config()

    def run(self):
        if self.force is not None:
            if self.force == 'all':
                force_run = [ f for f in range(self.nfolds) ]
            else:
                force_run = self.force
        else:
            force_run = [ f for f in range(self.nfolds) if self.config['classification']['status'][f] != "DONE" and
                                                           self.config['representation']['status'][f] == "DONE" ]
        
        for f in tqdm(force_run, desc=f"Running on folds ({self.representation.dname})", disable=self.silence, position=2):
            fold = self.representation.get_fold(f)
            #try:
            del self.classifier

            t_train = time()
            t_measure = time()
            self.classifier = self._Classifier_class_(**self.init_params)
            self.config['classification']['time_init'][f] = time() - t_measure

            if self.hyperparam_search is not None:
                t_measure = time()
                result = self.hyperparam_search.run(self.classifier, fold)
                self.config['classification']['time_hyperparam_search'][f] = time() - t_measure
                self.config['classification']['best_params'][f] = result.best_params
                self.classifier.set_params( **result.best_params )

            t_measure = time()
            self.classifier.fit(fold.X_train, fold.y_train)
            self.config['classification']['time_fit'][f] = time() - t_measure
            self.config['classification']['time_train'][f] = time() - t_train

            t_measure = time()
            y_pred = self.classifier.predict( fold.X_test )
            self.config['classification']['time_pred_test'][f] = time() - t_measure

            self.config['classification']['y_pred_test'][f] = y_pred
            self.config['classification']['y_true_test'][f] = fold.y_test

            if self.predict_proba:
                t_measure = time()
                y_pred_proba = self.classifier.predict_proba( fold.X_test )
                self.config['classification']['time_pred_test'][f] = time() - t_measure
                self.config['classification']['y_pred_proba_test'] = y_pred_proba
            
            self.config['classification']['status'][f] = "DONE"
            self.save_config()
            """except KeyboardInterrupt:
                self.save_config()
                break
            except Exception as e:
                self.config['status'][f] = traceback.format_exc()
                continue"""
        del self.classifier

    def save_config(self):
        save_json(path.join(self.output_path, 'result.json'), self.config)
