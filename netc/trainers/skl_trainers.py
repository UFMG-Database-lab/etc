from .trainers import Trainect
from ..utils.builder import from_descriptor


class SklTrainer(Trainect):
    def __init__(self, tname, descriptor):
        super(SklTrainer, self).__init__(tname, descriptor)
    def train_model(self, model, fold):
        X = fold.X_train
        y = fold.y_train
        if fold.with_val:
            X += fold.X_val
            y += fold.y_val
        model.fit(X, y)
        return {  }
class CVSklTrainer(SklTrainer):
    def __init__(self, tname, descriptor):
        super(CVSklTrainer, self).__init__(tname, descriptor)

    def init_model(self, fold, output_dir: str = None):
        from sklearn.model_selection import GridSearchCV
        submodel = from_descriptor(self.descriptor["skmodel"])

        model = GridSearchCV(submodel, self.descriptor["hyperparams"], **self.descriptor["gridparams"])
        return model
    def train_model(self, model, fold):
        X = fold.X_train
        y = fold.y_train
        if fold.with_val:
            X += fold.X_val
            y += fold.y_val
        model.fit(X, y)
        return { 'best_params': model.best_params_,  'cv': model.cv_results_ }