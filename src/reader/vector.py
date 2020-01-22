#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''Edit from DrQA'''

import torch
import logging

from collections import Counter
from src import num_docs

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- 
# VECTORIZE
# -----------------------------------------------------------------------------
    

def vectorize(ex, model, single_answer=False):
    
    '''Torchify a single example.'''
    
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.tensor([word_dict[w] for w in ex['document']], dtype=torch.long)
    question = torch.tensor([word_dict[w] for w in ex['question']], dtype=torch.long)

    # Create extra features vector
    if len(feature_dict) > 0:
        size = (len(ex['document']), len(feature_dict))
        features = torch.zeros(size)
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
    
    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.tensor(1, dtype=torch.long).fill_(ex['answers'][0][0])
        end = torch.tensor(1, dtype=torch.long).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, ex['id']
    

def vectorize1(ex, model, single_answer=False):
    
    '''Torchify a single example.'''
    
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.tensor([word_dict[w] for w in ex['document']], dtype=torch.long)
    question = torch.tensor([word_dict[w] for w in ex['question']], dtype=torch.long)
    
    # Create extra features vector
    if len(feature_dict) > 0:
        size = (len(ex['document']), len(feature_dict))
        features = torch.zeros(size)
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    return document, features, question, ex['id']


def vectorize_with_doc(ex, index, model, single_answer=False, docs_tmp=None):

    docs = []
    for i in range(0, num_docs):
        j = i % len(docs_tmp)

        docs_tmp[j]['answer'] = ex['answer'][0]
        docs_tmp[j]['id'] = index

        docs.append(vectorize1(docs_tmp[j], model, single_answer))
    
    return {"qa": ex, "docs": docs}


# ----------------------------------------------------------------------------- 
# BATCHIFY
# -----------------------------------------------------------------------------

    
def batchify(batch):
    '''Gather a batch of individual examples into one batch.'''
    
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    ids = [ex[-1] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    size = (len(docs), max_length)
    x1 = torch.zeros(size, dtype=torch.long)
    x1_mask = torch.empty(size, dtype=torch.bool).fill_(True)
    
    if features[0] is None:
        x1_f = None
    else:
        size = (len(docs), max_length, features[0].size(1))
        x1_f = torch.zeros(size)
    
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    size = (len(questions), max_length)
    x2 = torch.zeros(size, dtype=torch.long)
    x2_mask = torch.empty(size, dtype=torch.bool).fill_(True)
    
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            y_s = torch.cat([ex[3] for ex in batch])
            y_e = torch.cat([ex[4] for ex in batch])
        else:
            y_s = [ex[3] for ex in batch]
            y_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids


def batchify1(batch):
    '''Gather a batch of individual examples into one batch.'''

    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    ids = [ex[-1] for ex in batch]
    
    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    size = (len(docs), max_length)
    x1 = torch.zeros(size, dtype=torch.long)
    x1_mask = torch.empty(size, dtype=torch.bool).fill_(True)
    
    if features[0] is None:
        x1_f = None
    else:
        size = (len(docs), max_length, features[0].size(1))
        x1_f = torch.zeros(size)
    
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    size = (len(questions), max_length)
    x2 = torch.zeros(size, dtype=torch.long)
    x2_mask = torch.empty(size, dtype=torch.bool).fill_(True)
    
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
        
    # Return without targets
    return x1, x1_f, x1_mask, x2, x2_mask, ids


def batchify_with_docs(batch_list):
    
    res = []
    for i in range(num_docs):
        batch = []
        for ex in batch_list:
            batch.append(ex['docs'][i])
        res.append(batchify1(batch))
    
    return res
