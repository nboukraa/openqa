#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=spacy.npz'
    ),
}

''' 
    'Wikipedia Extractor' home page:
        http://medialab.di.unipi.it/wiki/Wikipedia_Extractor

    wikipedia/docs.db is available here:
        https://dl.fbaipublicfiles.com/drqa/docs.db.gz

    wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=spacy.npz is obtained by running:
        python scripts/retriever/build_tfidf.py wikipedia/docs.db wikipedia/ --ngram 2 --tokenizer simple --hash 16777216 
'''

def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)

from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker

