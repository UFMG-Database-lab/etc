from .descriptors.cls_desc import DESC_CLS
from .descriptors.dst_desc import DESC_DST
from .experiments.experiments import Experiment

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', nargs='+', default=DESC_DST.keys(),
                    help='Datasets to run.', choices=DESC_DST.keys())

parser.add_argument('-m', '--method', nargs='+', default=['tfidfsvm', 'bert-skl'],
                    help='Methods to run.', choices=DESC_CLS.keys())

parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('-s', '--save', action='store_true')

args = parser.parse_args()
print(args)

datasets = [ DESC_DST[d] for d in args.dataset ]
trainers = [ DESC_CLS[m] for m in args.method ]

exp = Experiment( dst_descs=datasets, trnr_descs=trainers, force=args.force, save_model=args.save  )
exp.run()
