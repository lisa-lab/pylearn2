#! /usr/env/bin python
# -*- coding: utf-8 -*-
"""
Datasets introduced in:

  Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.
  Richard Socher, ASlexPerelygin, Jean Wu, Jason Chuang, Christopher Manning, 
  Andrew Ng and Christopher Potts. Conference on Empirical Methods in Natural
  Language Processing (EMNLP 2013).
"""

__authors__ = "Trung Huynh"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__ = "3-clause BSD license"
__contact__ = "trunghlt@gmail.com"

import os
import csv
import urllib2
import zipfile
import logging
from collections import Counter
from tempfile import NamedTemporaryFile

import numpy as np
try:
    import scipy
    from distutils.version import StrictVersion
    if StrictVersion(scipy.__version__) < StrictVersion('0.13.0'):
        raise ImportError("Scipy version has to be >= 0.13.0")
except ImportError:
    import warnings
    warnings.warn("Couldn't import scipy")
try:
    from scipy.sparse import coo_matrix
except ImportError:
    warnings.warn("Couldn't import scipy.sparse")

from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets.text.text_dataset import TextDataset


log = logging.getLogger(__name__)


class StanfordSentimentTreebank(TextDataset):
    """ Load Stanford Sentiment Treebank """

    def __init__(self, which_set):
        sents, scores = _StanfordSentimentTreebankFactory.get(which_set)

    @property
    def vocabulary(self):
        return _StanfordSentimentTreebankFactory.vocabulary

    @property
    def docs(self):
        return _StanfordSentimentTreebankFactory.sents


class _StanfordSentimentTreebankFactory(object):
    """ Helper to load Stanford Sentiment Treebank """
    ORIGIN_URL = 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'
    DATA_ROOT = preprocess('${PYLEARN2_DATA_PATH}')
    DATA_PATH = os.path.join(DATA_ROOT, 'stanfordSentimentTreebank')
    SET_MAP = {'all': 0, 'train': 1, 'valid': 3, 'test': 2}
    SET_SIZE = {'all': 11855, 'train': 8544, 'valid': 1101, 'test': 2210}

    sents, sparse_sents, scores, splits, vocabulary = None, None, None, None, None

    @classmethod
    def get(cls, which_set):
        """ return respective dataset from cached data """
        assert which_set in ['all', 'train', 'valid', 'test']
        if not cls.__data_exists():
            cls.__download()
        if cls.sparse_sents is None or cls.scores is None:
            try:
                cls.__load()
            except IOError, err:
                logging.error("Can't load one of the data file. Set logging "
                              "level of {} with DEBUG and reload the dataset to "
                              "see what is wrong".format(__name__))
                raise IOError('Data corruption: {}'.format(err))
        set_id = cls.SET_MAP.get(which_set)
        idx = slice(None, None) if set_id == 0 else cls.splits == set_id
        X, y = cls.sparse_sents.tocsr()[idx], cls.scores[idx]
        assert X.shape[0] == y.shape[0], "Sentences and scores are misaligned"
        assert X.shape[0] == cls.SET_SIZE[which_set], \
            "{} has {} sentences, expected {}".format(which_set, X.shape[0],
                                                      cls.SET_SIZE[which_set])
        return X, y

    @classmethod
    def __data_exists(cls):
        fnames = ['.', 'datasetSplit.txt', 'dictionary.txt', 'SOStr.txt',
                  'sentiment_labels.txt', 'datasetSentences.txt']
        for fname in fnames:
            if not os.path.exists(os.path.join(cls.DATA_PATH, fname)):
                return False
        return True

    @classmethod
    def __load(cls):
        """ Load and cache data """
        splits = []
        with open(os.path.join(cls.DATA_PATH, 'datasetSplit.txt')) as f:
            reader = csv.reader(f, quotechar='"')
            reader.next()
            for row in reader:
                splits.append(int(row[1]))
            f.close()
        cls.splits = np.asarray(splits)

        phrases = {}
        with open(os.path.join(cls.DATA_PATH, 'dictionary.txt')) as f:
            reader = csv.reader(f, delimiter='|', quotechar='"')
            for phrase, id in reader:
                phrases[phrase] = int(id)
            f.close()

        sents, vocabulary, data, i, j = [], {}, [], [], []
        sent_split_path = os.path.join(cls.DATA_PATH, 'SOStr.txt')
        with open(sent_split_path) as f:
            reader = csv.reader(f, delimiter='|', quotechar='"')
            for sent_id, row in enumerate(reader):
                token_ids = []
                for token in row:
                    if token not in vocabulary:
                        vocabulary[token] = len(vocabulary)
                    token_ids.append(vocabulary[token])
                sents.append(token_ids)
                counter = Counter(token_ids)
                for token_id, count in counter.iteritems():
                    data.append(count)
                    i.append(sent_id)
                    j.append(token_id)
            f.close()
        cls.vocabulary = vocabulary
        cls.sents = sents
        cls.sparse_sents = coo_matrix((data, (i, j)))

        phrase_scores = []
        label_path = os.path.join(cls.DATA_PATH, 'sentiment_labels.txt')
        with open(label_path) as f:
            reader = csv.reader(f, delimiter='|', quotechar='"')
            reader.next()
            for sent_id, score in reader:
                phrase_scores.append(float(score))
        phrase_scores = np.asarray(phrase_scores)

        sent_path = os.path.join(cls.DATA_PATH, 'datasetSentences.txt')
        sent_scores = []
        with open(sent_path) as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            reader.next()
            for row in reader:
                # Fix doubled encoding
                sent = row[1].decode('utf-8').encode('latin-1')
                # Replace brackets by its original symbols
                sent = sent.replace('-LRB-', '(').replace('-RRB-', ')')
                sent_scores.append(phrase_scores[phrases[sent]])
            f.close()
        cls.scores = np.asarray(sent_scores)

    @classmethod
    def __download(cls):
        """ Download original data and unzip it to data path  """
        response = urllib2.urlopen(cls.ORIGIN_URL)
        content = response.read()
        with NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.close()
            log.debug('Downloaded content has been saved to {}'.format(f.name))
            with open(f.name) as g:
                zfile = zipfile.ZipFile(g)
                for name in zfile.namelist():
                    log.debug(
                        'Decompressing {} to {}'.format(name, cls.DATA_ROOT))
                    zfile.extract(name, cls.DATA_ROOT)
                g.close()


if __name__ == '__main__':
    # This is only for test purposes
    StanfordSentimentTreebank('all')
    StanfordSentimentTreebank('train')
    StanfordSentimentTreebank('test')
    StanfordSentimentTreebank('valid')
