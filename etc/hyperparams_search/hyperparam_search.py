from multiprocessing import cpu_count
from sklearn.base import BaseEstimator
from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import clone
import numpy as np




class HyperparamResult(object):
    def __init__(self, name_method, best_params, all_results={}):
        self.name_method = name_method
        self.best_params = best_params
        self.all_results = {}

class HyperparamSearch(BaseEstimator):
    def __init__(self, name_method):
        self.name_method = name_method
    def run(self, classifier, fold):
        raise NotImplementedError('HyperparamSearch is an abstract class.')

class GridSearchHyperparamSearch(HyperparamSearch):
    def __init__(self, parameters, scoring, cv=None, njobs=cpu_count()):
        super().__init__("GridSearchHyperparamSearch")
        self.parameters = parameters
        self.scoring = scoring
        self.cv = cv
        self.njobs = njobs
        self.clf = None
    def run(self, classifier, fold):
        self.clf = GridSearchCV(estimator=classifier, param_grid=self.parameters,
                            scoring=self.scoring, n_jobs=self.njobs, cv=self.cv)
        self.clf.fit(fold.X_train, fold.y_train)
        result = HyperparamResult(self.name_method, self.clf.best_params_, all_results=self.clf.cv_results_)
        return result

class RandomSearchHyperparamSearch(HyperparamSearch):
    def __init__(self, parameters, scoring, cv=None, njobs=cpu_count(), n_iter=10):
        super().__init__("RandomSearchHyperparamSearch")
        self.parameters = parameters
        self.scoring = scoring
        self.cv = cv
        self.njobs = njobs
        self.n_iter = n_iter
        self.clf = None
    def run(self, classifier, fold):
        self.clf = RandomizedSearchCV(estimator=classifier, param_distributions=self.parameters,
                            scoring=self.scoring, n_jobs=self.njobs, cv=self.cv, n_iter=self.n_iter)
        self.clf.fit(fold.X_train, fold.y_train)
        result = HyperparamResult(self.name_method, self.clf.best_params_, all_results=self.clf.cv_results_)
        
        return result

class BayesianHyperparamSearch(HyperparamSearch):
    def __init__(self, parameters, scoring, cv=None, njobs=cpu_count(), n_iter=10, random_state=42):
        super().__init__("BayesianHyperparamSearch")
        self.parameters = parameters
        self.scoring = scoring
        self.cv = cv
        self.njobs = njobs
        self.n_iter = n_iter
        self.clf = None
        self.random_state = np.random.RandomState(random_state)

    def __hyperopt_train_test(self, hyperparameters):
        classifier = clone(self.model)
        classifier.set_params(**hyperparameters)

        cv_metrics = cross_val_score(estimator=classifier, X=self.X, y=self.y, scoring=self.scoring, cv=self.cv,
                                     n_jobs=self.n_jobs, verbose=self.verbose)
        avg_metric = cv_metrics.mean()

        return avg_metric

    def __f(self, hyperparameters):
        avg_metric = self.__hyperopt_train_test(hyperparameters)
        loss_info = {"loss": -avg_metric, "status": STATUS_OK}

        return loss_info

    def run(self, classifier, fold):
        self.model = classifier
        self.X = fold.X_train
        self.y = fold.y_train
        best_parameters = fmin(fn=self.__f, space=self.parameters, algo=tpe.suggest, max_evals=self.n_iter,
                               rstate=self.random_state, verbose=self.verbose)

        best_parameters = space_eval(self.parameters, best_parameters)
        result = HyperparamResult(self.name_method, best_parameters)
        return result