from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from .trainers import Trainect
from ..utils.builder import from_descriptor

class HuggTrainer(Trainect):
    def __init__(self, tname, descriptor):
        super(HuggTrainer, self).__init__(tname, descriptor)
        self.model_name = descriptor['model_name']
    def init_model(self, fold, output_dir):
        print('initing model...')
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=fold.nclass, max_length=256)
        print('initing tknz...')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding=True, truncation=True, max_length=256)
        print('initing dataset...')
        fold = fold.to('hugging', tokenizer=tokenizer)
        calls = []
        if 'callbacks' in self.descriptor:
            calls = list(map(from_descriptor, self.descriptor['callbacks']))
        trainer = Trainer(
            model=model.to(self.descriptor['device'] if 'device' in self.descriptor else 'cuda:0'),
            args=TrainingArguments(**self.descriptor['training_args'],
                                    output_dir=output_dir),
            train_dataset=fold["train"],
            eval_dataset=fold["val"],
            tokenizer=tokenizer,
            callbacks=calls
        )
        return (model, tokenizer, trainer)

    def train_model(self, model, fold):
        (model, tokenizer, trainer) = model
        return { "train_output": trainer.train() }
        
    def predict(self, model, X):
        (model, tokenizer, trainer) = model
        return trainer.predict(X)
