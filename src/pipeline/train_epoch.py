#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''Main OpenQA training script.'''

import torch
import logging

from src import num_docs

from . import DocReader
from . import has_answer

from . import validate_with_doc

from src.reader import config
from src.reader import utils

# GPU usage
from GPUtil import showUtilization as gpu_usage

# Logger
logger = logging.getLogger()

# Global answer mapping
HasAnswer_Map = {}

# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------

def init_from_scratch(args, train_docs):
    '''New model, new data, new dictionary.'''
    
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args)
    logger.info('Num features = {}'.format(len(feature_dict)))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict_docs(args, train_docs)

    logger.info('Num words = {}'.format(len(word_dict)))

    # Initialize model
    model = DocReader(config.get_model_args(args), word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop with mode == 'all'
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, exs_with_doc, 
          docs_by_question):
    '''Run through one epoch of model training with the provided data loader.'''
    
    # Initialize meters and timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    
    # Run one epoch
    global HasAnswer_Map
    
    update_step = 0
    
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, ex_id = ex[0].size(0), ex[-1]
        
        # Display GPU usage statitstics every <display_stats> iterations
        show_stats = (args.show_cuda_stats and 
                      (idx % args.display_stats == args.display_stats -1))
        
        if (idx not in HasAnswer_Map):
            HasAnswer_list = []
            
            for idx_doc in range(0, num_docs):
                
                HasAnswer = []
                for i in range(batch_size):
                    
                    idx_doc_i = idx_doc %len(docs_by_question[ex_id[i]])
                    answer = exs_with_doc[ex_id[i]]['answer']
                    document = docs_by_question[ex_id[i]][idx_doc_i]['document']
                    
                    # ---------------------------------------------------------
                    # Looking for the answer in the document...
                    # ---------------------------------------------------------
                    HasAnswer.append(has_answer(args, answer, document))
                    # ---------------------------------------------------------
                HasAnswer_list.append(HasAnswer)
            
            HasAnswer_Map[idx] = HasAnswer_list
        
        else:
            HasAnswer_list = HasAnswer_Map[idx]

        # Initializing weights and sampling indices...
        weights = torch.tensor([1.0 for idx_doc in range(0, num_docs)])
        idx_random = torch.multinomial(weights, int(num_docs))

        HasAnswer_list_sample = []
        ex_with_doc_sample = []
        
        for idx_doc in idx_random:
            HasAnswer_list_sample.append(HasAnswer_list[idx_doc])
            ex_with_doc_sample.append(ex_with_doc[idx_doc])

        l_list_doc = []
        r_list_doc = []        
        for idx_doc in idx_random:
            
            l_list = []
            r_list = []
            for i in range(batch_size):
                if HasAnswer_list[idx_doc][i][0]:
                    l_list.append(HasAnswer_list[idx_doc][i][1])
                else:
                    l_list.append((-1,-1))
            
            l_list_doc.append(l_list)
            r_list_doc.append(r_list)
        
        # Generating predictions...
        pred_s_list_doc = []
        pred_e_list_doc = []
        tmp_top_n = 1
        
        # CUDA memory before forward pass
        txt_cuda(show_stats, 'before forward pass')
            
        for idx_doc in idx_random:
            ex = ex_with_doc[idx_doc]
            pred_s, pred_e, pred_score = model.predict(ex, top_n=tmp_top_n)
            
            pred_s_list = []
            pred_e_list = []
            for i in range(batch_size):
                pred_s_list.append(pred_s[i].tolist())
                pred_e_list.append(pred_e[i].tolist())

            pred_s_list_doc.append(torch.tensor(pred_s_list, dtype=torch.long))
            pred_e_list_doc.append(torch.tensor(pred_e_list, dtype=torch.long))
        
        # CUDA memory before backpropagation
        txt_cuda(show_stats, 'before backpropagation')
        
        # ---------------------------------------------------------------------
        # Updating (one epoch)...
        # ---------------------------------------------------------------------
        train_loss.update(*model.update_with_doc(
                update_step, ex_with_doc_sample, 
                pred_s_list_doc, pred_e_list_doc, tmp_top_n, 
                l_list_doc, r_list_doc, HasAnswer_list_sample))
        # ---------------------------------------------------------------------
        update_step = (update_step + 1) % 4
        # ---------------------------------------------------------------------
        
        # CUDA memory after backpropagation
        txt_cuda(show_stats, 'after backpropagation')
        if show_stats: gpu_usage()
        
        # Resetting...
        if idx % args.display_iter == 0:
            
            txt = 'train: Epoch = {} | iter = {}/{} | loss = {:.2f} | '
            txt+= 'elapsed time = {:.2f} (s)'
            logger.info(txt.format(
                    global_stats['epoch'], idx, len(data_loader), 
                    train_loss.avg, global_stats['timer'].time()))
            
            train_loss.reset()
        
        # Validation...
        if show_stats:
            with torch.no_grad():
                validate_with_doc(
                        args, data_loader, model, global_stats, exs_with_doc, 
                        docs_by_question, mode='train')
    
    logger.info('-' * 100) 
    txt = 'train: Epoch {} done. Time for epoch = {:.2f} (s)' 
    logger.info(txt.format(global_stats['epoch'], epoch_time.time()))
    logger.info('-' * 100)
    
    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] +1)



# ------------------------------------------------------------------------------
# Train loop with mode == 'selector'
# ------------------------------------------------------------------------------


def pretrain_selector(args, data_loader, model, global_stats, exs_with_doc, 
                      docs_by_question):
    '''Run through one epoch of model training with the provided data loader.'''
    
    # Initialize meters and timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    
    # Run one epoch
    global HasAnswer_Map
    
    tot_ans = 0
    tot_num = 0
    
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, ex_id = ex[0].size(0), ex[-1]
        
        # Update the answer mapping 
        # with starting and ending positions if an answer is found
        if (idx not in HasAnswer_Map):
            
            HasAnswer_list = []
            for idx_doc in range(0, num_docs):
            
                HasAnswer = []
                for i in range(batch_size):
                    
                    idx_doc_i = idx_doc %len(docs_by_question[ex_id[i]])
                    answer = exs_with_doc[ex_id[i]]['answer']
                    document = docs_by_question[ex_id[i]][idx_doc_i]['document']
                    
                    # ---------------------------------------------------------
                    # Looking for the answer in the document...
                    # [positions are ** ignored ** at this stage]
                    # ---------------------------------------------------------
                    bool_has, _ = has_answer(args, answer, document)
                    HasAnswer.append((bool_has,))
                    # ---------------------------------------------------------
                    
                HasAnswer_list.append(HasAnswer)
            HasAnswer_Map[idx] = HasAnswer_list
        
        else:
            HasAnswer_list = HasAnswer_Map[idx]
        
        # Update counters
        for idx_doc in range(0, num_docs):
            for i in range(batch_size):
                
                tot_ans += int(HasAnswer_list[idx_doc][i][0])
                tot_num += 1
        
        # Randomly sample the dataset to fit the model's input size
        weights = torch.tensor([1.0 for idx_doc in range(0, num_docs)])
        idx_random = torch.multinomial(weights, int(num_docs))

        HasAnswer_list_sample = []
        ex_with_doc_sample = []
        for idx_doc in idx_random:
            HasAnswer_list_idx_doc = [HasAnswer_list[idx_doc][i][0] 
            for i in range(batch_size)]
            HasAnswer_list_sample.append(HasAnswer_list_idx_doc)
            ex_with_doc_sample.append(ex_with_doc[idx_doc])
        
        HasAnswer_list_sample = torch.tensor(
                HasAnswer_list_sample, dtype=torch.long)
        
        # ---------------------------------------------------------------------
        # Updating train loss...
        # ---------------------------------------------------------------------
        train_loss.update(*model.pretrain_selector(
                ex_with_doc_sample, HasAnswer_list_sample))
        # ---------------------------------------------------------------------
        
        # Resetting...
        if idx % args.display_iter == 0:
            
            txt = 'train: Epoch = {} | iter = {}/{} | loss = {:.2f} | '
            txt+= 'elapsed time = {:.2f} (s)'
            logger.info(txt.format(
                    global_stats['epoch'], idx, len(data_loader), 
                    train_loss.avg, global_stats['timer'].time()))
            
            txt = 'tot_ans: {} | tot_num: {} | tot_ans/tot_num: {:.1f} (%)'
            logger.info(txt.format(
                    tot_ans, tot_num, tot_ans*100.0/tot_num))
            
            train_loss.reset()
    
    logger.info('-' * 100)
    txt = 'tot_ans: {} | tot_num: {}'
    logger.info(txt.format(tot_ans, tot_num))
    txt = 'train: Epoch {} done. Time for epoch = {:.2f} (s)'
    logger.info(txt.format(global_stats['epoch'], epoch_time.time()))
    logger.info('-' * 100)


# ------------------------------------------------------------------------------
# Train loop with mode == 'reader'
# ------------------------------------------------------------------------------


def pretrain_reader(args, data_loader, model, global_stats, exs_with_doc, 
                    docs_by_question):
    '''Run through one epoch of model training with the provided data loader.'''
    
    # Initialize meters and timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    
    logger.info('pretrain_reader')
    
    # Run one epoch
    global HasAnswer_Map
    
    count_ans = 0
    count_tot = 0
    
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, ex_id = ex[0].size(0), ex[-1]
        
        if (idx not in HasAnswer_Map):
            
            HasAnswer_list = []
            for idx_doc in range(0, num_docs):
            
                HasAnswer = []
                for i in range(batch_size):
                    
                    idx_doc_i = idx_doc %len(docs_by_question[ex_id[i]])
                    answer = exs_with_doc[ex_id[i]]['answer']
                    document = docs_by_question[ex_id[i]][idx_doc_i]['document']
                    
                    # Looking for the answer in the document...
                    # ---------------------------------------------------------
                    # Here we do care about the presence/absence of answers
                    # AND their positions in the documents
                    # ---------------------------------------------------------
                    HasAnswer.append(has_answer(args, answer, document))
                    # ---------------------------------------------------------
                    
                HasAnswer_list.append(HasAnswer)
            HasAnswer_Map[idx] = HasAnswer_list
        
        else:
            HasAnswer_list = HasAnswer_Map[idx]
        
        # Forward pass for the batch...
        for idx_doc in range(0, num_docs):
            
            l_list = []
            r_list = []
            
            # Forward pass for the batch...
            pred_s, pred_e, pred_score = model.predict(
                    ex_with_doc[idx_doc], top_n=1)
            
            for i in range(batch_size):
                
                if HasAnswer_list[idx_doc][i][0]:
                    count_ans += int(HasAnswer_list[idx_doc][i][0])
                    count_tot += 1
                    
                    # Store recorded answers' positions in a list
                    l_list.append(HasAnswer_list[idx_doc][i][1])
                
                else:
                    # Store the most answers' predicted positions
                    l_list.append([(int(pred_s[i][0]),int(pred_e[i][0]))])
            
            # -----------------------------------------------------------------
            # Model update: weights are adjusted so as to minimize the loss 
            # function / reducing inconsistencies between predicted and actual 
            # answer positions 
            # -----------------------------------------------------------------
            train_loss.update(*model.update(ex_with_doc[idx_doc], 
                    l_list, r_list, HasAnswer_list[idx_doc])) 
            # -----------------------------------------------------------------
            
        # Resetting train loss...
        if idx % args.display_iter == 0:
            
            txt = 'train: Epoch = {} | iter = {}/{} | loss = {:.2f} | '
            txt+= 'elapsed time = {:.2f} (s)'
            logger.info(txt.format(
                    global_stats['epoch'], idx, len(data_loader), 
                    train_loss.avg, global_stats['timer'].time()))
            
            train_loss.reset()
            
            txt = 'count_ans: {} | count_tot: {} | count_ans/count_tot: {:.2f} (%)'
            logger.info(txt.format(
                    count_ans, count_tot, 100.0*count_ans/(count_tot+1)))

    logger.info('-' * 100)
    txt = 'train: Epoch {} done. Time for epoch = {:.2f} (s)'
    logger.info(txt.format(global_stats['epoch'], epoch_time.time()))
    logger.info('-' * 100)


# ------------------------------------------------------------------------------
# CUDA usage
# ------------------------------------------------------------------------------


def txt_cuda(show, txt):
    if show:
        txt = 'CUDA memory ' + txt + ':\t {0:.2f} allocated, {0:.2f} cached'
        logger.info(txt.format(torch.cuda.memory_allocated() /1e9, 
                               torch.cuda.memory_cached() /1e9))
