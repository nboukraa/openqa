#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''Edit from OpenQA'''

import os
import json
import subprocess
import torch
import logging
import regex as re

from src import tokenizers
from src import sys_dir

from src.reader import utils
from src.retriever.utils import normalize

from multiprocessing.util import Finalize

#import ast
#import pprint

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Word tokenization
# --------------------------------------------------------------------------

''' ####################################################################### '''
#from src.pipeline import PROCESS_TOK
#PROCESS_TOK = None
''' ####################################################################### '''


tokenizers.set_default('corenlp_classpath', sys_dir+'/data/stanford-corenlp/*')

#tok_class = tokenizers.get_class('corenlp')
tok_class = tokenizers.get_class('spacy')
tok_opts = {}
PROCESS_TOK = tok_class(**tok_opts)
Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    
def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------


def read_data(filename, keys, res_len=-1):
    res = []
    step = 0
    for line in open(filename):
        
        ''' ############################################################### '''
        # similar amendment made in load_data_with_doc from src.reader.utils
        #if (res_len > 0) and (step > res_len): break
        #if step > 999: break
        ''' ############################################################### '''
        
        data = json.loads(line)
        
        # Available datasets: 'searchqa', 'quasart' and 'unftriviaqa'
        ''' ############################################################### '''
        if ('squad' in filename or 'webquestions' in filename): # not applicable
            answer = [tokenize_text(a).words() for a in data['answer']]
        else:
            if ('CuratedTrec' in filename):
                answer = data['answer']
            else:
                answer = [tokenize_text(a).words() for a in data['answers']]
        ''' ############################################################### '''
        
        question = ' '.join(tokenize_text(data['question']).words())
        
        res.append({'answer': answer, 'question': question})
        step+=1
        
    return res



def set_defaults(args):
    '''Make sure the commandline arguments are initialized properly.'''
    # Check critical files exist
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime('%Y%m%d-') + str(uuid.uuid4())[:8]

    # Set log and model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    
    return args


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


# Returns True if text t contains the passed answer 
# and yields a list of starting and ending positions where the answer was found
def has_answer(args, answer, t):
    global PROCESS_TOK
    
    text = []
    for i in range(len(t)):
        text.append(t[i].lower())
    
    res_list = []
    ''' ################################################################### '''
    if (args.dataset == 'CuratedTrec'): # not applicable
        try:
            ans_regex = re.compile('(%s)' %answer[0], 
                                   flags=re.IGNORECASE + re.UNICODE)
        except:
            return False, res_list
        
        paragraph = ' '.join(text)
        answer_new = ans_regex.findall(paragraph)
        
        for a in answer_new:
            single_answer = normalize(a[0])
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    res_list.append((i, i+len(single_answer)-1))
    
    else: # args.dataset == 'searchqa', 'quasart' or 'unftriviaqa'
        
        for a in answer:
            single_answer = ' '.join(a).lower()
            single_answer = normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    res_list.append((i, i+len(single_answer)-1))
    ''' ################################################################### '''
    
    if (len(res_list)>0):
        return True, res_list
    else:
        return False, res_list


# Similarity score
def set_sim(answer, prediction):

    ground_truths = []
    for a in answer:
        ground_truths.append(' '.join([w for w in a]))

    res = utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths)
    return res

def split_doc(doc):
    '''Given a doc, split it into chunks (by paragraph).'''
    GROUP_LENGTH = 0
    curr = []
    curr_len = 0
    for split in re.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > GROUP_LENGTH:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)


def eval_accuracies(pred_s, target_s, pred_e, target_e):
    '''An unofficial evalutation helper.
    Compute exact start/end/complete match accuracies for a batch.
    '''
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_s):
        target_s = [[e] for e in target_s]
        target_e = [[e] for e in target_e]

    # Compute accuracies from targets
    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matches
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        # End matches
        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100



# ------------------------------------------------------------------------------
# GPU Usage
# ------------------------------------------------------------------------------


def get_gpu_memory_map():
    """Get the current gpu usage.
    Credits: 
        https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/
    4?u=nabil_b
        https://unix.stackexchange.com/a/358990/388017
    
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
        All options can be found with: nvidia-smi --help-query-gpu
    """
    
    nvidia_options = [
            'timestamp', 'utilization.gpu', 'memory.used', 
            'memory.free', 'power.draw', 'temperature.gpu']
    
    # run nvidia-smi with '--format=csv' to view headers and units 
    nvidia_otions_with_units = [
            'timestamp', 'utilization.gpu [%]', 'memory.used [MiB]', 
            'memory.free [MiB]', 'power.draw [W]', 'temperature [C]']
    
    query = subprocess.check_output(
        [
            'nvidia-smi', 
            #'-l 1', # updates every second as a list
            '--query-gpu=' + ','.join(nvidia_options),
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # output
    query = query.strip().split(', ') #.split('\n')
    nvidia_map = dict(zip(nvidia_otions_with_units, query))
    
    return nvidia_map