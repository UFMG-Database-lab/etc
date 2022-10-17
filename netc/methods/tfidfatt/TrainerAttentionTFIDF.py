from ...trainers.trainers import Trainect

class TrainerAttentionTFIDF(Trainect):
    def __init__(self, tname, descriptor):
        super(TrainerAttentionTFIDF, self).__init__(tname, descriptor)

    def train_model(self, model, fold):
        self.t_eval['train.tknz'].tick
        model.tknz.fit(fold.X_train, fold.y_train)
        self.t_eval['train.tknz'].tick

        self.t_eval['train.model'].tick
        results_ = model.train(fold)
        self.t_eval['train.model'].tick

        return results_

    def save_model(self, output_path, model):
        model.save(output_path)