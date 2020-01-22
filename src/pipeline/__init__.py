#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#from src import sys_dir
#from src import MODEL_DIR, EMBED_DIR

#from ..reader import config
#from ..reader import utils
from ..reader import vector
from ..reader.model import DocReader
from ..reader.vector import vectorize, batchify
from ..reader.vector import vectorize_with_doc, batchify_with_docs
from ..reader.data import ReaderDataset, ReaderDataset_with_Doc

from .utils import PROCESS_TOK

from .utils import str2bool
from .utils import set_defaults
from .utils import tokenize_text
from .utils import read_data
from .utils import has_answer
from .utils import set_sim

from .config import add_main_args
from .config import add_interface_args

from .validation_epoch import validate_with_doc

from .train_epoch import train
from .train_epoch import init_from_scratch
from .train_epoch import pretrain_reader
from .train_epoch import pretrain_selector
from .train_epoch import txt_cuda

