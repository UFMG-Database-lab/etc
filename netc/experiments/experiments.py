from itertools import product
from os import path
from ..utils.base import create_path, seed_everything
from ..utils.builder import from_descriptor
from tqdm.auto import tqdm
from ..datasets.dataset import Dataset


class Experiment(object):
    def __init__(self, dst_descs, trnr_descs, output_path='~/.etc/', save_model=False, force=False, seed=42):
        super(Experiment, self).__init__()
        self.datasets = [ Dataset(**desc) for desc in dst_descs ]
        self.trnr_descs = trnr_descs # [ from_descriptor(desc) for desc in trnr_descs ]
        self.N = len(self.datasets)*len(self.trnr_descs)
        self.force = force
        self.output_path = path.abspath(path.join(path.expanduser(output_path), "results"))
        self.save_model = save_model
        self.seed = seed
        self.init_experiments()

    def init_experiments(self):
        for dst in self.datasets:
            create_path(path.join(self.output_path, dst.dname))

    def run(self):
        for dst in tqdm(self.datasets, desc=f"Dataset...", position=4):
            for fold in dst:
                for desc in tqdm(self.trnr_descs, desc=f"Method...", position=3):
                    seed_everything(self.seed)
                    trnr = from_descriptor(desc)
                    foutput_path = path.join(self.output_path, dst.dname, trnr.tname, f"fold-{fold.foldname}", str(fold.fold_idx))
                    trnr.run( fold, foutput_path, save_model = self.save_model, force = self.force )
