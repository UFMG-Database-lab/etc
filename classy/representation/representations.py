# -*- coding: utf-8 -*-

import numpy as np
import argparse
from os import path

from dataset import Dataset
from codes.utils import save_splits_ids, load_splits_ids, create_splits
from codes.utils import read_dataset, load_json, save_json, save_vectorize, create_path
from codes.utils import dump_svmlight_file_gz, get_array, is_jsonable
from codes.generic_estimator import GenericVectorizer

from datetime import datetime

from time import time
from tqdm import tqdm

def load_json_configs(json_files):
    configs_to_return = []
    for json_file in json_files:
        json_list = load_json(json_file)
        configs_to_return.extend(json_list)
    return configs_to_return


"""
def create_default( ntrain, ntest ):
     with open('split_dafault.csv','w') as fil_out:
             train_ids  = range(ntrain)
             test_ids   = range(ntrain,ntrain+ntest)
             train_ids_str = ' '.join( list(map(str, train_ids)) )
             test_ids_str = ' '.join( list(map(str, test_ids)) )
             fil_out.write( ';'.join([train_ids_str, test_ids_str]) )
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split input dataset into k folds of cross-validation and create representations.')
    
    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument('-d', '--datasetdir', required=True, type=str, nargs='+', help='Dataset path (For more info: readme.txt)')

    general_args = parser.add_argument_group('Representations arguments')
    general_args.add_argument('-ir', '--input_repr', type=str, nargs='+', default=[], help=f'[Optional]')

    general_args = parser.add_argument_group('Configurations arguments')
    general_args.add_argument('--silence', action="store_false", help=f'Silence the bar.')
    general_args.add_argument('-dsm','--dont_save_model', action="store_false", help=f'To do not save the Vectorizer.')
    general_args.add_argument('-s', '--seed', type=int, default=42, help=f'Seed to randomic generation.')
    general_args.add_argument('-f', '--nfolds', type=str, default='10', help=f'Number of fold to build (if the folds are already made, the splits will be used).')
    general_args.add_argument('-o', '--output', type=str, default=path.join('..','..','representations'), help=f'Path to the output directory (to save the splits and representations).')

    #general_args = parser.add_argument_group('Representations commands')
    #general_args.add_argument('-a', '--add', type=str, help=f'Add the predefined .')
    #general_args.add_argument('-r', '--remove', type=str, default='', help=f'.')
    
    args = parser.parse_args()

    for datasetpath in tqdm(args.datasetdir, desc="Running datasets", total=len(args.datasetdir), position=0, disable=not args.silence):
        config_output = {}

        # Create dataset_name
        dataset = Dataset(datasetpath)
        dname = dataset.dname
        config_output['input_dataset'] = path.abspath(datasetpath)
        config_output['dname'] = dname
        #config_output['nfolds'] = args.nfolds

        vectorizers_configs = load_json_configs(args.input_repr)
        for vectorizer_conf in tqdm(vectorizers_configs, total=len(vectorizers_configs), position=1, desc=f'Running repr ({dname})', disable=(not len(vectorizers_configs) or not args.silence)):
            config_output['vectorizer'] = vectorizer_conf['class_module_path']
            config_output['vectorizer_alias'] = vectorizer_conf['alias']
            config_output['save_model'] = args.dont_save_model
            config_output['creation_vectorizer_date'] = str(datetime.now())
            config_output['time_vectorizer_init'] = []
            config_output['time_vectorizer_fit'] = []
            config_output['time_vectorizer_train'] = []
            config_output['time_vectorizer_test'] = []
            for (f, fold) in tqdm(enumerate(dataset.get_fold_instances(args.nfolds, with_val=False)), position=2,desc=f"Running folds ({vectorizer_conf['alias']})", disable=not args.silence ):
                # Build vectorizer
                # Build vectorizer
                time_delta = time()
                gv = GenericVectorizer(**vectorizer_conf)
                config_output['time_vectorizer_init'].append( time() - time_delta )

                # Get GenericVectorizer unique name
                config_output['name_vectorizer_method'] = str(gv)

                # Get GenericVectorizer params
                # Convert to str if the param is non-serializable
                config_output['params_vectorizer'] = { k: v if is_jsonable(v) else str(v) for (k,v) in gv.get_params().items() }
                
                # Fit the vectorizer
                time_delta = time()
                gv.fit( fold.X_train, fold.y_train )
                config_output['time_vectorizer_fit'].append( time() - time_delta )

                # Build the representation of the train
                time_delta = time()
                X_train_repr = gv.transform( fold.X_train )
                config_output['time_vectorizer_train'].append( time() - time_delta )

                # Build the representation of the test
                time_delta = time()
                X_test_repr = gv.transform( fold.X_test )
                config_output['time_vectorizer_test'].append( time() - time_delta )

                # define directory 
                config_dir = path.join(args.output, dname, f'fold-{args.nfolds}', config_output['name_vectorizer_method'])
                create_path(config_dir)

                # Save the outputs
                dump_svmlight_file_gz(X_train_repr, list(map(int,fold.y_train)), path.join(config_dir, 'train%d.gz' % f))
                dump_svmlight_file_gz(X_test_repr, list(map(int,fold.y_test)), path.join(config_dir, 'test%d.gz' % f))

                # Save the vectorizer
                if config_output['save_model']:
                    save_vectorize(gv.est, path.join(config_dir, 'vectorizer-%d.pkl' % f))
                
                del gv
            # Save the configs
            config_output['nfolds'] = f+1
            save_json(path.join(config_dir, 'configuration.json'), config_output)
        # END: Run representations