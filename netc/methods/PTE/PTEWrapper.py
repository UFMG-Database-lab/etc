TMP_DIR= '/tmp/'

"""
"""

from subprocess import call
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from os import path, remove
import numpy as np


PATH_TO_EXEC = path.join(path.dirname(path.relpath(__file__)), 'exec')

LOG_FILE = path.join(PATH_TO_EXEC, 'LOG.txt')

DATA2W = './'+ path.join(PATH_TO_EXEC, 'data2w')
DATA2DL= './'+ path.join(PATH_TO_EXEC, 'data2dl')
PTE= './'+ path.join(PATH_TO_EXEC, 'pte')
INFER= './'+ path.join(PATH_TO_EXEC, 'infer')

def filter_text(doc):
    doc = doc.lower()
    for c in ["'",'"',".",",",":","!","#","?","$","%","^","&","*","(",")"]: # Unefficient way V0
        doc = doc.replace(c, ' ')
    return doc
def save_file(filename, content, end=''):
    with open(filename, 'w') as filout:
        for line in content:
            filout.write(str(line))
            filout.write(end)
def load_representation(filename):
    data = []
    with open(filename) as filin:
        for idx_row, line in enumerate(filin.readlines()[1:]):
            data.append([ float(part) for part in line.split()[1:] ])
    return np.array(data)
def remove_if_exists(filename):
    if path.exists(filename):
        remove(filename)

from multiprocessing import cpu_count
class PTEVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=cpu_count(), d=300, samples=300, w=5, mindf=2):
        self.d = d
        self.w = w
        self.mindf = mindf
        self.samples = samples
        self.n_jobs = n_jobs
        self.vocabulary_ = dict()
        self.key = None
    def uniquekey(self):
        from datetime import datetime
        dt = datetime.now()
        return str(dt.microsecond)
    def fit(self, X,y=None, *kwargs):
        self.key = self.uniquekey()
        
        TMP_IN_TRAIN_FILE = f'{TMP_DIR}{self.key}train_in'
        TMP_IN_LABEL = f'{TMP_DIR}{self.key}train_label'
        TMP_IN_EMBBEDING = f'{TMP_DIR}{self.key}train_embedding'
        TMP_OUT_TRAIN_FILE = f'{TMP_DIR}{self.key}train_out'

        remove_if_exists(TMP_IN_TRAIN_FILE)
        remove_if_exists(f"{TMP_DIR}{self.key}ww.net")
        remove_if_exists(f"{TMP_DIR}{self.key}words.node")
        remove_if_exists(f"{TMP_DIR}{self.key}lw.net")
        remove_if_exists(f"{TMP_DIR}{self.key}labels.node")
        remove_if_exists(f"{TMP_DIR}{self.key}dw.net")
        remove_if_exists(f"{TMP_DIR}{self.key}docs.node")
        remove_if_exists(f"{TMP_DIR}{self.key}text.node")
        remove_if_exists(f"{TMP_DIR}{self.key}text.hin")
        remove_if_exists(f"{TMP_DIR}{self.key}word.emb")
        remove_if_exists(TMP_OUT_TRAIN_FILE)
        

        X2 = list(map(filter_text, X))
        save_file(TMP_IN_TRAIN_FILE, X2)
        save_file(TMP_IN_LABEL, y, end='\n')

        call(f"{DATA2W} -text {TMP_IN_TRAIN_FILE} \
            -output-ww {TMP_DIR}{self.key}ww.net \
            -output-words {TMP_DIR}{self.key}words.node \
            -window {self.w}  \
            -threads {self.n_jobs} \
            -min-count {self.mindf} > {LOG_FILE}", shell=True)

        call(f"{DATA2DL} ./text2hin/data2dl \
            -text {TMP_IN_TRAIN_FILE} \
            -label {TMP_IN_LABEL}  \
            -threads {self.n_jobs} \
            -output-lw {TMP_DIR}{self.key}lw.net \
            -output-labels {TMP_DIR}{self.key}labels.node \
            -output-dw {TMP_DIR}{self.key}dw.net \
            -output-docs {TMP_DIR}{self.key}docs.node \
            -min-count {self.mindf} >> {LOG_FILE}", shell=True)

        call(f"cat {TMP_DIR}{self.key}ww.net {TMP_DIR}{self.key}dw.net {TMP_DIR}{self.key}lw.net > {TMP_DIR}{self.key}text.hin", shell=True)
        call(f"cat {TMP_DIR}{self.key}words.node {TMP_DIR}{self.key}docs.node {TMP_DIR}{self.key}labels.node > {TMP_DIR}{self.key}text.node", shell=True)
        
        call(f"{PTE} -nodes {TMP_DIR}{self.key}text.node \
            -words {TMP_DIR}{self.key}words.node \
            -hin {TMP_DIR}{self.key}text.hin \
            -output {TMP_DIR}{self.key}word.emb \
            -binary 1 \
            -negative 5 \
            -size {self.d} \
            -samples {self.samples} \
            -threads {self.n_jobs} >> {LOG_FILE}", shell=True)
        return self
    def transform(self, X):
        if self.key is None:
            raise Exception('Not fitted yet!')
        
        TMP_IN_TRAIN_FILE = f'{TMP_DIR}{self.key}train_in'
        TMP_IN_LABEL = f'{TMP_DIR}{self.key}train_label'
        TMP_IN_EMBBEDING = f'{TMP_DIR}{self.key}train_embedding'
        TMP_OUT_TRAIN_FILE = f'{TMP_DIR}{self.key}train_out'
        
        remove_if_exists(TMP_IN_TRAIN_FILE)
        X2 = list(map(filter_text, X))
        save_file(TMP_IN_TRAIN_FILE, X2)
        call(f"{INFER} -infer {TMP_IN_TRAIN_FILE} \
            -vector {TMP_DIR}{self.key}word.emb \
            -output {TMP_OUT_TRAIN_FILE} \
            -debug 2 -binary 0 >> {LOG_FILE}", shell=True)
        X3 = load_representation(TMP_OUT_TRAIN_FILE)
        return X3