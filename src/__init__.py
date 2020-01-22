#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging
from pathlib import PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('Only supports Python 3.5 or higher.')

# ------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------

# Number of paragraphs contained to extract from each document
# (50 in the original research paper)
num_docs = 30 

# Number of guesses to display
display_num = int(num_docs/3) # == 10

# Probability of extracting answer given a question and a paragraph
# (1) Max: we assume that only one token in the paragraph indicates the correct answer
# (2) Sum: we regard all tokens matched to the correct answer equally
type_max = True

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------

sys_dir = '/home/nboukraa/Documents/projet-personnel/open-qa'
MODEL_DIR = sys_dir + '/models/'
EMBED_DIR = sys_dir + '/data/embeddings/'
embed_name = 'glove.840B.300d.txt'

DATA_DIR = (
    os.getenv('DRQA_DATA') or
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)

# ------------------------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------------------------

logger = logging.getLogger()
logger.setLevel(logging.INFO)

txtfmt = '%(asctime)s: [ %(message)s ]' 
datefmt = '%m/%d/%Y %I:%M:%S %p'
fmt = logging.Formatter(txtfmt, datefmt)

console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# CHILDREN
# ------------------------------------------------------------------------------

from . import tokenizers
from . import reader
from . import retriever
from . import pipeline
