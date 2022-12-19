
from .trainers import Trainect

class TrainerETC(Trainect):
    def __init__(self, tname, descriptor):
        super(TrainerETC, self).__init__(tname, descriptor)

    def train_model(self, model, fold):
        self.t_eval['train.tknz'].tick
        model.tknz.fit(fold.X_train, fold.y_train)
        self.t_eval['train.tknz'].tick

        self.t_eval['train.model'].tick
        results_ = model.fit( X_train=fold.X_train, y_train=fold.y_train, X_val=fold.X_val, y_val=fold.y_val)
        self.t_eval['train.model'].tick

        return results_

    def save_model(self, output_path, model):
        model.save(output_path)