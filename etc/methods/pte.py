
TMP_DIR= '/tmp/'
TMP_IN_TRAIN_FILE = f'{TMP_DIR}train_in'
TMP_IN_LABEL = f'{TMP_DIR}train_label'
TMP_IN_EMBBEDING = f'{TMP_DIR}train_embedding'
TMP_OUT_TRAIN_FILE = f'{TMP_DIR}train_out'

"""
"""

from subprocess import call
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from os import path, remove
import numpy as np


PATH_TO_EXEC = path.join(path.dirname(path.relpath(__file__)), 'exec')

LOG_FILE = path.join(PATH_TO_EXEC, 'LOG.txt')

DATA2W = './'+ path.join(PATH_TO_EXEC, 'commands', 'data2w')
DATA2DL= './'+ path.join(PATH_TO_EXEC, 'commands', 'data2dl')
PTE= './'+ path.join(PATH_TO_EXEC, 'commands', 'pte')
INFER= './'+ path.join(PATH_TO_EXEC, 'commands', 'infer')

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

class PTEVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=-1, d=100, samples=300, w=5, mindf=1):
        self.d = d
        self.w = w
        self.mindf = mindf
        self.samples = samples
        self.n_jobs = n_jobs
        self.vocabulary_ = dict()

        
    def fit(self, X,y=None, *kwargs):
        remove_if_exists(TMP_IN_TRAIN_FILE)
        remove_if_exists(f"{TMP_DIR}ww.net")
        remove_if_exists(f"{TMP_DIR}words.node")
        remove_if_exists(f"{TMP_DIR}lw.net")
        remove_if_exists(f"{TMP_DIR}labels.node")
        remove_if_exists(f"{TMP_DIR}dw.net")
        remove_if_exists(f"{TMP_DIR}docs.node")
        remove_if_exists(f"{TMP_DIR}text.node")
        remove_if_exists(f"{TMP_DIR}text.hin")
        remove_if_exists(f"{TMP_DIR}word.emb")
        remove_if_exists(TMP_OUT_TRAIN_FILE)
        

        X2 = list(map(filter_text, X))
        save_file(TMP_IN_TRAIN_FILE, X2)
        save_file(TMP_IN_LABEL, y, end='\n')

        call(f"{DATA2W} -text {TMP_IN_TRAIN_FILE} \
            -output-ww {TMP_DIR}ww.net \
            -output-words {TMP_DIR}words.node \
            -window {self.w} \
            -min-count {self.mindf} > {LOG_FILE}", shell=True)

        call(f"{DATA2DL} ./text2hin/data2dl \
            -text {TMP_IN_TRAIN_FILE} \
            -label {TMP_IN_LABEL} \
            -output-lw {TMP_DIR}lw.net \
            -output-labels {TMP_DIR}labels.node \
            -output-dw {TMP_DIR}dw.net \
            -output-docs {TMP_DIR}docs.node \
            -min-count {self.mindf} >> {LOG_FILE}", shell=True)

        call(f"cat {TMP_DIR}ww.net {TMP_DIR}dw.net {TMP_DIR}lw.net > {TMP_DIR}text.hin", shell=True)
        call(f"cat {TMP_DIR}words.node {TMP_DIR}docs.node {TMP_DIR}labels.node > {TMP_DIR}text.node", shell=True)
        
        call(f"{PTE} -nodes {TMP_DIR}text.node \
            -words {TMP_DIR}words.node \
            -hin {TMP_DIR}text.hin \
            -output {TMP_DIR}word.emb \
            -binary 1 \
            -negative 5 \
            -size {self.d} \
            -samples {self.samples} \
            -threads {self.n_jobs} >> {LOG_FILE}", shell=True)
        return self
    def transform(self, X):
        #raise Exception("Not implemented yet!")
        remove_if_exists(TMP_IN_TRAIN_FILE)
        X2 = list(map(filter_text, X))
        save_file(TMP_IN_TRAIN_FILE, X2)
        call(f"{INFER} -infer {TMP_IN_TRAIN_FILE} \
            -vector {TMP_DIR}word.emb \
            -output {TMP_OUT_TRAIN_FILE} \
            -debug 2 -binary 0 >> {LOG_FILE}", shell=True)
        X3 = load_representation(TMP_OUT_TRAIN_FILE)
        return X3
