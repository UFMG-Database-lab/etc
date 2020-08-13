
from nltk.corpus import stopwords
from .stop_words import ENGLISH_STOP_WORDS_17
import re

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import scipy.sparse as sp
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES

replace_patterns = [
    ('<[^>]*>', ''),                                    # remove HTML tags
    ('(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2'),
    ('(\D)\d(\D)', '\\1ParsedOneDigit\\2'),
    ('(\D)\d\d(\D)', '\\1ParsedTwoDigits\\2'),
    ('(\D)\d\d\d(\D)', '\\1ParsedThreeDigits\\2'),
    ('(\D)\d\d\d\d(\D)', '\\1ParsedFourDigits\\2'),
    ('(\D)\d\d\d\d\d(\D)', '\\1ParsedFiveDigits\\2'),
    ('(\D)\d\d\d\d\d\d(\D)', '\\1ParsedSixDigits\\2'),
    ('\d+', 'ParsedDigits')
]

compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]

def generate_preprocessor(replace_patterns):
    compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]
    def preprocessor(text):
        # For each pattern, replace it with the appropriate string
        for pattern, replace in compiled_replace_patterns:
            text = re.sub(pattern, replace, text)
        text = text.lower()
        return text
    return preprocessor

generated_patters=generate_preprocessor(replace_patterns)

def preprocessor(text):
    # For each pattern, replace it with the appropriate string
    for pattern, replace in compiled_replace_patterns:
        text = re.sub(pattern, replace, text)
    text = text.lower()
    return text

class TFIDFRepresentation(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        #super().__init__("TFIDF", use_validation=False )
        self.stopwords=stopwords
        self.preprocessor=preprocessor
        self.input=input
        self.encoding=encoding
        self.decode_error=decode_error
        self.strip_accents=strip_accents
        self.lowercase=lowercase
        self.tokenizer=tokenizer
        self.analyzer=analyzer
        self.token_pattern=token_pattern
        self.ngram_range=ngram_range
        self.max_df=max_df
        self.min_df=min_df
        self.max_features=max_features
        self.vocabulary=vocabulary
        self.binary=binary
        self.dtype=dtype
        self.norm=norm
        self.use_idf=use_idf
        self.smooth_idf=smooth_idf
        self.sublinear_tf=sublinear_tf

        self.vectorize = TfidfVectorizer(input=self.input, encoding=self.encoding, decode_error=self.decode_error,
            strip_accents=self.strip_accents, lowercase=self.lowercase,
            preprocessor=self.preprocessor, tokenizer=self.tokenizer, analyzer=self.analyzer,
            stop_words=self.stopwords, token_pattern=self.token_pattern,
            ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df,
            max_features=self.max_features, vocabulary=self.vocabulary, smooth_idf=self.smooth_idf, binary=self.binary,
            dtype=self.dtype)

    def fit(self, X_train, y_train):
        self.vectorize.fit(X_train)
        return self

    def transform(self, X):
        return self.vectorize.transform(X)

class TFRepresentation(CountVectorizer):
    def __init__(self, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stopwords, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)
        self.stopwords = stopwords

# FIX (TODO): Arrumar os parâmetros de inicialização das versões Stemmed dos vectorizer
"""
class StemmedTFICFVectorizer(TFICFVectorizer):
    def __init__(self, lang, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
                preprocessor=preprocessor, 
                stop_words=stopwords, input=input, encoding=encoding,
                 decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
                 tokenizer=tokenizer, analyzer=analyzer,
                 token_pattern=token_pattern,
                 ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                 max_features=max_features, vocabulary=vocabulary, binary=binary,
                 dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                 sublinear_tf=sublinear_tf)
        self.lang = lang
        self.stopwords = stopwords
        from nltk.stem.snowball import SnowballStemmer
        from nltk import word_tokenize
        self.stemmer = SnowballStemmer(self.lang)
    def build_analyzer(self):
        analyzer = super(TFICFVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))

class StemmedTFIDFVectorizer(TFIDFVectorizer):
    def __init__(self, lang, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
                preprocessor=preprocessor, 
                stop_words=stopwords, input=input, encoding=encoding,
                 decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
                 tokenizer=tokenizer, analyzer=analyzer,
                 token_pattern=token_pattern,
                 ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                 max_features=max_features, vocabulary=vocabulary, binary=binary,
                 dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                 sublinear_tf=sublinear_tf)
        self.lang = lang
        self.stopwords = stopwords
        from nltk.stem.snowball import SnowballStemmer
        from nltk import word_tokenize
        self.stemmer = SnowballStemmer(self.lang)
    def build_analyzer(self):
        analyzer = super(TFIDFVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))

class StemmedTFVectorizer(TFVectorizer):
    def __init__(self, lang, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
                preprocessor=preprocessor, 
                stop_words=stopwords, input=input, encoding=encoding,
                 decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
                 tokenizer=tokenizer, analyzer=analyzer,
                 token_pattern=token_pattern,
                 ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                 max_features=max_features, vocabulary=vocabulary, binary=binary,
                 dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                 sublinear_tf=sublinear_tf)
        self.lang = lang
        self.stopwords = stopwords
        from nltk.stem.snowball import SnowballStemmer
        from nltk import word_tokenize
        self.stemmer = SnowballStemmer(self.lang)
    def build_analyzer(self):
        analyzer = super(TFVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))

"""