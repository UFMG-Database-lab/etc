
import argparse
from os import path
from base import Representation
from Executors import ExecutorClassifier
from tqdm import tqdm
from default_descriptors import default_cls, default_hyperparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end.')
    
    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument('-r', '--repr', required=True, type=str, nargs='+', help='Representation path (For more info: readme.txt)')

    general_args = parser.add_argument_group('Method descriptor arguments')
    general_args.add_argument('-i', '--input_method', type=str, nargs='+', default=[], help=f'[Optional] Input of the method descriptor.')
    general_args.add_argument('-hps', '--hyperparam_search', type=str, default=None, help=f'[Optional] Input of the hyperparam search.')

    general_args = parser.add_argument_group('Configurations arguments')
    general_args.add_argument('--silence', action="store_true", default=False, help=f'Silence the progress bar.')
    general_args.add_argument('-sm','--save_model', action="store_true", default=False, help=f'To save the Classifier object.')
    general_args.add_argument('-pp','--predict_proba', action="store_true", default=False, help=f'Save the predicted probabilities to all class.')
    general_args.add_argument('-s', '--seed', type=int, default=42, help=f'Seed to randomic generation.')
    general_args.add_argument('-o', '--output', type=str, default=path.join('..','..','..','etc','class'), help=f'Path to the output directory (to save classification results).')
    
    general_args.add_argument('-F', '--force', nargs='+', default=None, help=f'Force fold to (re)execute.')

    args = parser.parse_args()
    if args.force is not None:
        if 'all' in args.force:
            args.force = 'all'
        else:
            args.force = list(map(int, args.force))

    print(args)

    for reprpath in tqdm(args.repr, desc="Running on datasets/representation", disable=args.silence, position=1):
        repr_obj = Representation(reprpath)
        for method_descriptor in args.input_method:
            if method_descriptor in default_cls:
                method = default_cls[method_descriptor] ### TODO: PRECISO VOLTAR PARA VERSÃO DO SKLEARN (até pq n vamos ter validação)
            if args.hyperparam_search is not None and args.hyperparam_search in default_hyperparams:
                args.hyperparam_search = default_hyperparams[args.hyperparam_search]
                print(args.hyperparam_search)
            executor = ExecutorClassifier( repr_obj, args.output, method,
                save_model=args.save_model, predict_proba=args.predict_proba,
                force=args.force, hyperparam_search=args.hyperparam_search, silence=args.silence )
            executor.run()