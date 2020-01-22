#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''Interactive interface to the full OpenQA pipeline.'''

import torch
import argparse
import code
import time
import regex
import json

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from . import DocReader
from . import ReaderDataset_with_Doc

from . import set_defaults
from . import tokenize_text
from . import add_interface_args
from . import vectorize
from . import batchify, batchify_with_docs

from src import sys_dir
from src import logger
from src.retriever import get_class
from src.reader import config

from src import tokenizers
from src.retriever import DEFAULTS
from src.retriever import DocDB

from src import num_docs, display_num
from src.reader import utils

# Display loyout
import jsbeautifier
import prettytable
from termcolor import colored

opts = jsbeautifier.default_options()
opts.indent_size = 4



# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch and tokenize text
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None
PROCESS_CANDS = None

def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

''' def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text) '''

# Target size for squashing short paragraphs together.
# 0 = read every paragraph independently
# infty = read all paragraphs together
GROUP_LENGTH = 0


# ------------------------------------------------------------------------------
# Question answering
# ------------------------------------------------------------------------------


def process(args, question, ranker, model,
            openqa=True, candidates=None, top_n=1, n_docs=5, return_context=True):
    
    # timer
    t0 = time.time()
    
    # -------------------------------------------------------------------------
    # Document Ranker
    # -------------------------------------------------------------------------

    query = ' '.join(tokenize_text(question).words())
    doc_ids, doc_scores = ranker.closest_docs(query, k=n_docs)
    doc_idx = [str(ranker.get_doc_index(d)) for d in doc_ids]
    
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Index', 'Doc Score']
    )
    
    if return_context:
        for i in range(len(doc_ids)):
            table.add_row([i + 1, doc_ids[i], doc_idx[i], '{:.5g}'.format(doc_scores[i])])
        
        print(table)    
    
    # -------------------------------------------------------------------------
    # Processing of retrieved documents
    # -------------------------------------------------------------------------
   
    # Initializing tokenizers and document retrievers...
    logger.info('Initializing tokenizers and document retrievers...')
    tokenizer = 'spacy'
    tok_class = tokenizers.get_class(tokenizer)
    annotators = tokenizers.get_annotators_for_model(model)
    tok_opts = {'annotators': annotators}
    
    db_config = {'options': {'db_path': DEFAULTS['db_path']}} 
    db_class = db_config.get('class', DocDB)
    db_opts = db_config.get('options', {})
    
    fixed_candidates = candidates is not None
    num_workers = args.data_workers
    
    processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, fixed_candidates)
        )
    
    # Retrieve text from database.
    d2d_mapping = {did: didx for didx, did in enumerate(doc_ids)}
    doc_texts = processes.map(fetch_text, doc_ids)

    # Split and flatten documents. Maintain a mapping from doc (index in
    # flat list) to split (index in flat list).
    splits = []
    d2s_mapping = [] # doc to split mapping
    for text in doc_texts:
        
        d2s_mapping.append([len(splits), -1])
        for split in split_doc(text):
            splits.append(split)
        d2s_mapping[-1][1] = len(splits)   
    
    # Push through the tokenizers as fast as possible.
    query = [query]
    
    q_tokens = processes.map_async(tokenize_text, query).get()
    s_tokens = processes.map_async(tokenize_text, splits).get()

    # Group into structured example inputs.   
    examples = []
    qidx = 0
    
    for rel_didx, did in enumerate(doc_ids):
        
        start, end = d2s_mapping[d2d_mapping[did]]
        for sidx in range(start, end):
        
            if (len(q_tokens[qidx].words()) > 0 and
                    len(s_tokens[sidx].words()) > 0):
            
                examples.append({
                    'question': q_tokens[qidx].words(),
                    'document': s_tokens[sidx].words(),
                    'qlemma': q_tokens[qidx].lemmas(),
                    'lemma': s_tokens[sidx].lemmas(),
                    'pos': s_tokens[sidx].pos(),
                    'ner': s_tokens[sidx].entities(),
                    'id': (qidx, rel_didx, sidx),
                })
    
    logger.info('Reading {} paragraphs...'.format(len(examples)))
        
    # ---------------------------------------------------------------------
    # Document Reader
    # ---------------------------------------------------------------------
    
    if not openqa: 
        
        # ---------------------------------------------------------------------
        # Document Reader - DrQA
        # ---------------------------------------------------------------------
        
        # Build the batch and run it through the model
        batch_exs = batchify([vectorize(ex, model) for ex in examples])
        s, e, score = model.predict(batch_exs, candidates, top_n)
    
        # Retrieve the predicted spans
        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                span = s_tokens[i].slice(s[i][j], e[i][j] + 1).untokenize()
                predictions.append((span, score[i][j].item()))
            results.append(predictions)
        
        return sorted(results, key=lambda x: x[0][1], reverse=True)[:display_num]
    
    else:
        
        # -------------------------------------------------------------------------
        # Document Reader - OpenQA
        # -------------------------------------------------------------------------
        
        # Create a dummy example with an empty list of documents        
        query_exs_with_doc = read_data({'question': question, 'answers': ['']})
    
        # Load all examples into an output file        
        filename_query_docs = (sys_dir + '/data/datasets/' 
                               + args.dataset + '/query.json')
                
        with open(filename_query_docs, 'w+') as outfile:
            json.dump(examples, outfile, sort_keys = False)
        
        # Reload the examples...
        query_docs, _, _ = utils.load_data_with_doc(
                args, filename_query_docs)
                
        query_loader_with_doc = get_loader(
                args, query_exs_with_doc, model, query_docs)
        
        best_answer = predictions_with_doc(
                args, model, query_loader_with_doc, query_docs, 
                return_context=return_context, min_score=.75)
        
        print('\n')
        logger.info('Best answer: {} found in {:.2f} (s)'.format(
                best_answer, time.time() - t0))
               


# ------------------------------------------------------------------------------
# User interface
# ------------------------------------------------------------------------------


def user_interface(mode):
    
    # Model arguments
    parser = argparse.ArgumentParser(
            'OpenQA Document Reader',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    add_interface_args(parser, mode)
    config.add_model_args(parser)
    args = parser.parse_args()
    
    set_defaults(args)
    
    # CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU {})'.format(args.gpu))
    else:
        logger.info('Running on CPU only.')
    
    logger.info('Initializing pipeline...')
    
    # Load the tf-idf ranker
    logger.info('Initializing document ranker...')
    ranker = get_class('tfidf')()
    
    # Load a trained reader model
    logger.info('Loading selected model...')
    model = DocReader.load(args.model_file, new_args=args)
    
    model.network.eval()
    model.selector.eval()
    
    def ask(question=None, 
            candidates=None, top_n=1, n_docs=5, return_context=True):
        
        if question is None: 
            question = 'What is question answering?'
        
        return process(args, question, ranker, model, 
                       candidates=None, top_n=1, n_docs=5, return_context=True)
                
    # Interactive mode
    logger.info('-' * 100)
    logger.info('Entering interactice mode...')
    
    banner = '''
    Interactive OpenQA 
    >> ask(question, top_n=1, n_docs=5, return_context=True)
    >> ask() will yield the answer to 'What is question answering?'
    press Ctrl+D to leave
    '''
    
    code.interact(banner=banner, local=locals())



# ------------------------------------------------------------------------------
# Read paragrapahs to find the most likely answer
# ------------------------------------------------------------------------------

def predictions_with_doc(args, model, query_loader, query_docs, 
                         return_context=True, min_score=.75):
    ''' Predict the most likely answers from a batch of documents '''
    
    logger.info('predictions_with_doc')
    
    for idx, ex_with_doc in enumerate(query_loader):
        
        ex = ex_with_doc[0] # documents
        batch_size = ex[0].size(0) # batch size == 1 by construction
        ex_id = ex[-1] # identifiers
               
        # ---------------------------------------------------------------------
        # Document Selector
        # ---------------------------------------------------------------------
        
        # Get the most likely answers from the batch and see if the answer 
        # is actually in there           
        scores_doc_num = model.predict_with_doc(ex_with_doc)
        scores = [{} for i in range(batch_size)]
        
        # ---------------------------------------------------------------------
        # Document Reader
        # ---------------------------------------------------------------------
        # Update performance metrics (batch size == 1 by construction)
        for i in range(batch_size):

            _, indices = scores_doc_num[i].sort(0, descending=True)
        
            for idx_doc in indices: #range(0, num_docs):
                ex = ex_with_doc[idx_doc]
                pred_s, pred_e, pred_score = model.predict(
                        ex, top_n=display_num)

                idx_doc_i = idx_doc %len(query_docs[ex_id[i]])
                doc_text = query_docs[ex_id[i]][idx_doc_i]['document']
                
                max_display = min(len(pred_score[i]), display_num)
                
                # read the 10 best predicted answers
                for k in range(max_display): #range(display_num):
                    
                    try:
                        prediction = [doc_text[j] for j in range(pred_s[i][k], 
                                      pred_e[i][k]+1)]
                        prediction = ' '.join(prediction).lower()
                        
                        # update prediction scores
                        if (prediction not in scores[i]): 
                            scores[i][prediction] = 0
                        scores[i][prediction] += (pred_score[i][k] * 
                              scores_doc_num[i][idx_doc])

                        # Print relevant answers found in the passed document 
                        if (return_context and 
                            ((pred_score[i][k] > min_score) or 
                             (scores_doc_num[i][idx_doc] > min_score)) or
                             (scores[i][prediction] > min_score **2)):
                            
                            long_answer = (doc_text[:pred_s[i][k]] + 
                                           ['>>>'] + doc_text[pred_s[i][k]:pred_e[i][k]+1] + 
                                           ['<<<'] + doc_text[pred_e[i][k]+1:])
                            long_answer = ' '.join(long_answer).lower()
                            
                            str1 = '{:.2f}%'.format(scores_doc_num[i][idx_doc] * 100)
                            str2 = '{:.2f}%'.format(pred_score[i][k] * 100)
                            str3 = '{:.2f}%'.format(scores[i][prediction] * 100)
                            
                            best_guess = {'prediction score (selector)': str1,
                                          'prediction score (reader)': str2,
                                          'prediction score (pipeline)': str3,
                                          'long answer': long_answer}
                            
                            print(jsbeautifier.beautify(json.dumps(best_guess), opts))
    
                    except:
                        pass
        
            # Find the most likely short answer
            a = sorted(scores[i].items(), key=lambda d: d[1], reverse=True)
            #logger.info('Most likely answer scores ranked from best to poorest')
            #logger.info(a[:display_num])
            
            # Update performance metrics
            best_score = 0
            prediction = ''
            for key in scores[i]:
                if (scores[i][key] > best_score):
                    best_score = scores[i][key]
                    prediction = key
    
            assert(prediction == a[0][0])
            
    return {'Most likely answer': prediction}


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

def split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    
    curr = []
    curr_len = 0
    
    for split in regex.split(r'\n+', doc):
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


def get_loader(args, exs_with_doc, model, docs):
    """Return a pytorch data iterator for provided examples."""
    
    dataset_with_doc = ReaderDataset_with_Doc(
        exs_with_doc, model, docs, single_answer=False
        )
    
    sampler_with_doc = SequentialSampler(
        dataset_with_doc
        )
    
    loader_with_doc = DataLoader(
       dataset_with_doc,
       batch_size=args.test_batch_size,
       sampler=sampler_with_doc,
       num_workers=args.data_workers,
       collate_fn=batchify_with_docs,
       pin_memory=args.cuda,
       )
    
    return loader_with_doc


def read_data(ex):
    ''' This is expected the same effect on the input as read_data from 
    src.reader.utils ''' 
    res = []
    
    answer = [tokenize_text(a).words() for a in ex['answers']]
    question = ' '.join(tokenize_text(ex['question']).words())
    res.append({'answer': answer, 'question': question})        
    
    return res
