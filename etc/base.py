
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np
from urllib.parse import urljoin as urllibj
from glob import glob
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import gzip
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import io
import json
import re
from os import path, mkdir, remove
import pickle
from sklearn.feature_extraction.text import CountVectorizer

WithValFold = namedtuple('Fold', ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val'])
Fold = namedtuple('Fold', ['X_train', 'y_train', 'X_test', 'y_test'])




def dump_svmlight_file_gz(X,y,filename):
    with gzip.open(filename, 'w') as filout:
        dump_svmlight_file(X, y, filout, zero_based=False)
def filter_text(text, params=None):
    tfvector = CountVectorizer().build_tokenizer()
    return list( map(' '.join, map(tfvector, text)) )
def remove_if_exists(filename):
    if path.exists(filename):
        remove(filename)
def save_file(filename, content):
    with open(filename, 'w') as filout:
        for line in content:
            filout.write(line + '\n')
def create_path(path_to_create):
    path_to_create = path.abspath(path_to_create)
    paths = path_to_create.split(path.sep)
    complete_path = '/'
    for p in paths[1:]:
        complete_path = path.join(complete_path, p)
        if not path.exists(complete_path):
            mkdir( complete_path )
def read_lines(filename):
    with io.open(filename, newline='\n') as filin:
        return filin.readlines()
def load_json(path_json_file):
    if path_json_file is None:
        return []
    with open(path_json_file) as file_in:
        json_obj = json.load(file_in)
    return json_obj
def save_json(path_json_file, data):
    with open(path_json_file, 'w') as file_out:
        json.dump(data, file_out, cls=NumpyEncoder)
def is_jsonable(x):
    try:
        json.dumps(x, cls=NumpyEncoder)
        return True
    except (TypeError, OverflowError):
        return False
def urljoin(url, *args):
    for part in args:
        url = urllibj(url, part)
    return url
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
class Tokenizer():
    def __init__(self, vectorizer=None, maxlen=None, **kwargs):
        self.term2ix = { 'END': 0 }
        self.maxlen = maxlen
        if 'build_tokenizer' in dir(vectorizer):
            self.tokenizer = vectorizer
        else:
            self.tokenizer = CountVectorizer(**kwargs)

    def doc2termid(self, text):
        doc_by_id = []
        for i,term in enumerate(self.tokenize(text)):
            if self.maxlen is not None and i == self.maxlen:
                break
            termid = self.term2ix.setdefault( term, len(self.term2ix) )
            doc_by_id.append( termid )
        doc_by_id.append( 0 )
        return doc_by_id
    def transform(self, text):
        doc_by_id = []
        for i,term in enumerate(self.tokenize(text)):
            if self.maxlen is not None and i == self.maxlen:
                break
            if term not in self.term2ix:
                continue
            termid = self.term2ix[term]
            doc_by_id.append( termid )
        doc_by_id.append( 0 )
        return doc_by_id

    def tokenize(self, text):
        return self.tokenizer.build_tokenizer()(text)
    
    def get_ix2term(self):
        return [ k for k,v in sorted( self.term2ix.items(), key=lambda x: x[1] )]
    def get_term2ix(self):
        return self.term2ix
class Representation(object):
    def __init__(self, representationpath):
        self.representationpath = representationpath
        json_path = path.join(self.representationpath, 'result.json')
        self.config = load_json(json_path)
        self.nfold = self.config['nfolds']
        self.dname = self.config['dname']
        self.name_fold = self.config['name_fold']
        self.name_repr = self.config['representation']['name_method']

    def get_fold(self, f):
        
        filename_train = path.join(self.representationpath, f'train_{f}.gz')
        X_train, y_train = load_svmlight_file(filename_train, zero_based=False)

        filename_test  = path.join(self.representationpath, f'test_{f}.gz')
        X_test, y_test = load_svmlight_file(filename_test, n_features=X_train.shape[1], zero_based=False)

        fold = Fold(X_train, y_train, X_test, y_test)
        return fold

    def get_param(self, param):
        if param not in self.config:
            raise ValueError(f"Param '{param}' not found!")
        return self.config[param]   
class Dataset(object):
    def __init__(self, dname, dataset_path='~/', repo='http://homepages.dcc.ufmg.br/~vitormangaravite/', random_state=42, encoding='utf8'):
        super(Dataset, self).__init__()
        self.dname = dname.lower()
        dataset_path = path.expanduser(dataset_path)
        self.dataset_path = path.abspath(path.join(dataset_path, '.etc', 'datasets', dname))
        
        self.texts_filepath = path.join(self.dataset_path, 'texts.txt')
        self.score_filepath = path.join(self.dataset_path, 'score.txt')
        self.splits_path    = path.join(self.dataset_path, 'splits')
        if repo is not None:
            repo  =  path.join(repo, '')
            repo +=  '/'.join(['.etc', 'datasets', dname])

        self.repo = repo
        self.random_state = random_state
        self.encoding = encoding

        self._load_dataset_()
        self.split = {}

    def __str__(self):
        return f"<Dataset ({self.dname})>"
    def __repr__(self):
        return self.__str__()

    @property
    def ndocs(self):
        return len(self.y)

    @staticmethod
    def get_array(X, idxs):
        return [ X[idx] for idx in idxs ]
        
    def nfold(self, name_split):
        splits = self.__getitem__(name_split)
        return len( splits )

    def get_fold(self, f, name_split, with_val=True):
        splits = self.__getitem__(name_split)
        if f >= self.nfold(name_split):
            # Fold id not found!
            raise ValueError(f"Fold idx {f} not found in {name_split} splits.")
        fold = self._build_instances_(splits[f], with_val)
        return fold
    
    def __getitem__(self, name_split):
        name_split = str(name_split)
        if name_split not in self.available_splits:
            # Create name_split
            split = self._download_split_(name_split)
            self.split[name_split] = split if split is not None else self._create_splits_( int(name_split) )
            self.available_splits.add( name_split )
        if name_split not in self.split:
            # Load name_split
            self.split[name_split] = self._load_splits_(name_split)
            if len(self.split[name_split][0]) != 3:
                self.split[name_split]  = self._create_val_(self.split[name_split])
                self._save_split_(name_split, self.split[name_split])
        return self.split[name_split]
    
    def _download_split_(self, name_split):
        split_path = self.splits_path+'/split_%s.csv' % str(name_split)
        content = self._download_( split_path )
        if content is not None:
            save_file(split_path, content)
            return self._load_splits_(name_split)
        return None

    def _create_splits_(self, k):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        kf  = list(skf.split(self.texts, self.y))
        kf  = self._create_val_(kf)
        self._save_split_(k, kf)
        return kf

    def _create_val_(self, split):
        aux_split = []
        for (train_ids, test_ids) in split:
            try:
                train_idx_atual, val_idx_atual = train_test_split(train_ids,
                                                test_size=len(test_ids),
                                                stratify=Dataset.get_array(self.y, train_ids),
                                                random_state=self.random_state)
            except ValueError:
                train_idx_atual, val_idx_atual = train_test_split(train_ids,
                                                test_size=len(test_ids),
                                                random_state=self.random_state)
                
            aux_split.append( (train_idx_atual, val_idx_atual, test_ids) )
        return aux_split

    def _save_split_(self, name_split, splits):
        split_file = path.join(self.dataset_path, 'splits', f'split_{name_split}.csv')
        with open(split_file, 'w', encoding=self.encoding, errors='ignore') as fileout:
            for train_index, val_index, test_index in splits:
                train_str = ' '.join(list(map(str, train_index)))
                val_str   = ' '.join(list(map(str, val_index)))
                test_str  = ' '.join(list(map(str, test_index)))
                line = train_str + ';' + val_str + ';' + test_str + '\n'
                fileout.write(line)
    
    def _load_splits_(self, name_split):
        splits = []
        filepath = path.join(self.dataset_path, 'splits', f'split_{name_split}.csv')
        if not path.exists(filepath):
            raise FileNotFoundError(f"Split {name_split} not found!")
        with open(filepath, encoding=self.encoding, errors='ignore') as fileout:
            for line in fileout.readlines():
                fold = []
                for idx_part in line.split(';'):
                    index = list(map(int, idx_part.split()))
                    fold.append( index )
                
                splits.append( tuple(fold) )
        return splits


    def _download_n_save_(self, filepath, url):
        text = self._download_(url)
        if text is not None:
            with open(filepath, 'w', encoding=self.encoding) as fil_out:
                fil_out.write( text )
        else:
            raise Exception("File not found in %s!" % url)
            
    def _download_(self, url):
        import urllib.request
        from ssl import _create_unverified_context
        try:
            context  = _create_unverified_context()
            response = urllib.request.urlopen(url, context=context)
            data     = response.read()
            return data.decode(self.encoding)
        except:
            return None

    def _load_dataset_(self):
        if not path.exists(self.dataset_path) or \
            not path.exists(self.texts_filepath) or \
            not path.exists(self.score_filepath) or \
            not path.exists(self.splits_path):
            
            if not self.repo:
                raise Exception('Dataset %s not found!' % self.dname)

            create_path(self.splits_path)
            
            if not path.exists(self.texts_filepath):
                self._download_n_save_( self.texts_filepath, self.repo+'/texts.txt' )

            if not path.exists(self.score_filepath):
                self._download_n_save_( self.score_filepath, self.repo+'/score.txt' )


        self.texts = read_lines(self.texts_filepath)
        self.y = read_lines(self.score_filepath)
        self.y = list(map(int, self.y))
        splits_files = glob( path.join(self.splits_path, 'split_*.csv') )
        self.available_splits = set(map(lambda x: path.basename(x)[6:-4], splits_files ))
        self.nclass = len(set(self.y))
        self.split = dict()
    
    def _build_instances_(self, split, with_val):
        train_idx, val_idx, test_idx = split

        # Test values
        X_test  = Dataset.get_array( self.texts, test_idx )
        y_test  = Dataset.get_array( self.y, test_idx )

        if with_val:
            # Train values
            X_train = Dataset.get_array( self.texts, train_idx )
            y_train = Dataset.get_array( self.y, train_idx )

            # Validation values
            X_val   = Dataset.get_array( self.texts, val_idx )
            y_val   = Dataset.get_array( self.y, val_idx )

            fold = WithValFold( X_train, y_train, X_test, y_test, X_val, y_val )
        else:
            train_idx = sorted(np.concatenate([train_idx, val_idx]))

            # Train values
            X_train = Dataset.get_array( self.texts, train_idx )
            y_train = Dataset.get_array( self.y, train_idx )
            fold = Fold( X_train, y_train, X_test, y_test )
            
        return fold