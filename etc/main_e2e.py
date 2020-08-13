
import argparse
from os import path
from base import Dataset
from Executors import ExecutorE2E
from tqdm import tqdm
from default_descriptors import default_cls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end.')
    
    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument('-d', '--dataset', required=True, type=str, nargs='+', help='Dataset path (For more info: readme.txt)')

    general_args = parser.add_argument_group('Method descriptor arguments')
    general_args.add_argument('-i', '--input_method', type=str, nargs='+', default=[], help=f'[Optional] Input of the method descriptor.')

    general_args = parser.add_argument_group('Configurations arguments')
    general_args.add_argument('--silence', action="store_true", default=False, help=f'Silence the progress bar.')
    general_args.add_argument('-sm','--save_model', action="store_true", default=False, help=f'To save the Classifier object.')
    general_args.add_argument('-pp','--predict_proba', action="store_true", default=False, help=f'Save the predicted probabilities to all class.')
    general_args.add_argument('-s', '--seed', type=int, default=42, help=f'Seed to randomic generation.')
    general_args.add_argument('-f', '--nfolds', type=str, default='10', help=f'Name of fold to build (if the folds are already made, the splits will be used).')
    general_args.add_argument('-enc', '--encoding', type=str, default='utf8', help=f'Encoding to read/write dataset.')
    general_args.add_argument('-o', '--output', type=str, default=path.join('..','..','..','etc','class'), help=f'Path to the output directory (to save classification results).')
    
    general_args.add_argument('-F', '--force', nargs='+', default=None, help=f'Force fold to (re)execute.')

    args = parser.parse_args()
    if args.force is not None:
        if 'all' in args.force:
            args.force = 'all'
        else:
            args.force = list(map(int, args.force))

    print(args)

    for datasetpath in tqdm(args.dataset, desc="Running on datasets", disable=args.silence, position=1):
        dataset_obj = Dataset(datasetpath, random_state=args.seed, encoding=args.encoding)
        for method_descriptor in args.input_method:
            if method_descriptor in default_cls:
                method = default_cls[method_descriptor]
            executor = ExecutorE2E( dataset_obj, args.output, method, args.nfolds,
                hyperparam_search=None, save_model=args.save_model,
                predict_proba=args.predict_proba, force=args.force, silence=args.silence )
            executor.run()