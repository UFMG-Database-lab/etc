import argparse
from os import path
from tqdm import tqdm
import json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end.')
    
    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument('-r', '--result', required=True, type=str, nargs='+', help='The result paths.')

    general_args = parser.add_argument_group('Configurations arguments')
    general_args.add_argument('-T','--times', action="store_true", default=False, help=f'Show times.')
    general_args.add_argument('-a','--all', action="store_true", default=False, help=f'Show alls stats.')
    general_args.add_argument('-m','--metric', default=['f1_micro'], help=f'Metric.')


    args = parser.parse_args()

