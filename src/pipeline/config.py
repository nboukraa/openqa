#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model options for the training phase."""

import os
import sys
import logging

from src import DATA_DIR as DRQA_DATA
from src import MODEL_DIR, EMBED_DIR
from src import sys_dir, embed_name

sys.path.append(sys_dir)
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')

from src.pipeline import str2bool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


def add_main_args(parser, mode):
    '''Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    
    ---------------------------------------------------------------------------
    Step 1: pre-train the paragraph reader: 
        >>>  python main.py --batch-size 256 --model-name quasart_reader 
                            --num-epochs 10 --dataset quasart --mode reader
    ---------------------------------------------------------------------------
    Step 2: pre-train the paragraph selector: 
        >>>  python main.py --batch-size 64 --model-name quasart_selector 
                            --num-epochs 10 --dataset quasart --mode selector 
                            --pretrained models/quasart_reader.mdl
    ---------------------------------------------------------------------------
    Step 3: train the whole pipeline: 
        >>>  python main.py --batch-size 32 --model-name quasart_all 
                            --num-epochs 10 --dataset quasart --mode all 
                            --pretrained models/quasart_selector.mdl   
    
    '''
    
    parser.register('type', 'bool', str2bool)
    
    # Training moode
    if mode == 'reader':
        batch_size = 256
        test_batch_size = 64
        model_name = 'quasart_reader'
        pretrained = None
        num_epochs = 10
    
    elif mode == 'selector':
        batch_size = 64
        test_batch_size = 64
        model_name = 'quasart_selector'
        pretrained = 'models/quasart_reader.mdl'
        num_epochs = 10
    
    elif mode == 'all':
        batch_size = 32
        test_batch_size = 32
        model_name = 'quasart_all'
        pretrained = 'models/quasart_selector.mdl'
        num_epochs = 10
        
    else:
        raise RuntimeError('Unsupported training mode: {}'.format(mode))
    
    # -------------------------------------------------------------------------
    # Runtime environment
    # -------------------------------------------------------------------------
    runtime = parser.add_argument_group('Environment')
    
    runtime.add_argument('--dataset', 
                         type=str, default='quasart', #searchqa
                         help='Dataset: searchqa, quasart or unftriviaqa')
    runtime.add_argument('--mode', 
                         type=str, default=mode, #'all'
                         help='Train_mode: all, reader or selector')
    runtime.add_argument('--no-cuda', 
                         type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', 
                         type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--show-cuda-stats', 
                         type=int, default=False, #True for testing
                         help='Show CUDA statistics during execution')
    runtime.add_argument('--data-workers', 
                         type=int, default=12, #0 #1
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', 
                         type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', 
                         type=int, default=1012,
                         help=('Random seed (for reproducibility)'))
    runtime.add_argument('--num-epochs', 
                         type=int, default=num_epochs, #20
                         help='Train data iterations')
    runtime.add_argument('--batch-size', 
                         type=int, default=batch_size, #128
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', 
                         type=int, default=test_batch_size, #64
                         help='Batch size during validation/testing')

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------
    files = parser.add_argument_group('Filesystem')
    
    files.add_argument('--model-dir', 
                       type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', 
                       type=str, default=model_name,
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', 
                       type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--embed-dir', 
                       type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', 
                       type=str, default=embed_name,
                       help='Space-separated pretrained embeddings file')

    # Saving and loading
    save_load = parser.add_argument_group('Saving/Loading')
    
    save_load.add_argument('--checkpoint', 
                           type='bool', default=True, #False
                           help='Save model and optimizer state after each epoch')
    save_load.add_argument('--pretrained', 
                           type=str, default=pretrained,
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', 
                           type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    
    # -------------------------------------------------------------------------
    # Data preprocessing
    # -------------------------------------------------------------------------
    preprocess = parser.add_argument_group('Preprocessing')
    
    preprocess.add_argument('--uncased-question', 
                            type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', 
                            type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', 
                            type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    general = parser.add_argument_group('General')
    
    general.add_argument('--official-eval', 
                         type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', 
                         type=str, default='exact_match',
                         help='If using official evaluation: f1; else: exact_match')
    general.add_argument('--display-iter', 
                         type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--display-stats', 
                         type=int, default=200,
                         help='Display stats every <display_stats> epochs')
    general.add_argument('--sort-by-len', 
                         type='bool', default=True,
                         help='Sort batches by length for speed')



# ------------------------------------------------------------------------------
# Arguments to use in interactive mode.
# ------------------------------------------------------------------------------


def add_interface_args(parser, mode):
    '''Adds commandline arguments to use on interactive mode
    '''
    
    parser.register('type', 'bool', str2bool)

    # Evaluation
    if mode == 'query':
        model_name = 'quasart_all'
    else:
        raise NotImplementedError(mode)

    # -------------------------------------------------------------------------
    # Runtime environment
    # -------------------------------------------------------------------------
    runtime = parser.add_argument_group('Environment')
    
    runtime.add_argument('--dataset', 
                         type=str, default='quasart', #'searchqa',
                         help='Dataset: searchqa, quasart or unftriviaqa')
    runtime.add_argument('--no-cuda', 
                         type='bool', default=True,
                         help='CPU only for interactive mode')
    runtime.add_argument('--data-workers', 
                         type=int, default=12,
                         help='Number of subprocesses for CPU usage')
    runtime.add_argument('--random-seed', 
                         type=int, default=1012,
                         help=('Random seed (for reproducibility)'))
    runtime.add_argument('--test-batch-size', 
                         type=int, default=64,
                         help='Batch size during validation/testing')
    
    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------
    files = parser.add_argument_group('Filesystem')
    
    files.add_argument('--model-dir', 
                       type=str, default=MODEL_DIR,
                       help='Path to trained Document Reader model')
    files.add_argument('--model-name', 
                       type=str, default=model_name,
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--embed-dir', 
                       type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', 
                       type=str, default=embed_name,
                       help='Space-separated pretrained embeddings file')
    parser.add_argument('--tokenizer', 
                        type=str, default='spacy', #None
                        help=("Tokenizer to use (e.g. 'spacy')"))
    
    # -------------------------------------------------------------------------
    # Data preprocessing
    # -------------------------------------------------------------------------
    preprocess = parser.add_argument_group('Preprocessing')
    
    preprocess.add_argument('--uncased-question', 
                            type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', 
                            type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', 
                            type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    
    parser.add_argument('--candidate-file', 
                        type=str, default=None,
                        help=("List of candidates to restrict predictions to, "
                              "one candidate per line"))