#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Main OpenQA training and testing script.'''

import os
import sys
import logging
import argparse
import torch
import numpy as np
import json
import gc

from src import sys_dir
from src import logger, txtfmt

from src.reader import utils, vector, config, data
from src.reader import DocReader

from src.pipeline import set_defaults
from src.pipeline import read_data
from src.pipeline import train
from src.pipeline import add_main_args
from src.pipeline import init_from_scratch
from src.pipeline import pretrain_reader
from src.pipeline import pretrain_selector
from src.pipeline import validate_with_doc
from src.pipeline import txt_cuda

# GPU usage
from GPUtil import showUtilization as gpu_usage


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def load_train_evaluate_save(mode):
    
    # -------------------------------------------------------------------------   
    # PARSER
    # -------------------------------------------------------------------------   
    
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'OpenQA Question Answering Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_main_args(parser, mode)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)
        
    
    # -------------------------------------------------------------------------   
    # INITIALIZATIONS
    # -------------------------------------------------------------------------   
    
    # CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert(args.cuda)
    if args.cuda:
        torch.cuda.set_device(args.gpu) # no-op if args.gpu is negative
        torch.cuda.empty_cache()
    
    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
    
        logfile.setFormatter(txtfmt)
        logger.addHandler(logfile)
    
    logger.info('COMMAND: {}'.format(' '.join(sys.argv)))
    
    # GPU cleaning
    gc.collect()
    for obj in gc.get_objects():
        del obj
    torch.cuda.empty_cache()
    
    
    # --------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------   
    
    logger.info('-' * 100)
    logger.info('Load data files')
    
    dataset = args.dataset # == 'searchqa', 'quasart' or 'unftriviaqa'
    
    filename_train_docs = sys_dir+'/data/datasets/'+dataset+'/train.json' 
    filename_dev_docs = sys_dir+'/data/datasets/'+dataset+'/dev.json' 
    filename_test_docs = sys_dir+'/data/datasets/'+dataset+'/test.json' 
    filename_train = sys_dir+'/data/datasets/'+dataset+'/train.txt' 
    filename_dev = sys_dir+'/data/datasets/'+dataset+'/dev.txt' 
    filename_test = sys_dir+'/data/datasets/'+dataset+'/test.txt'
    
    train_docs, train_questions, train_len = utils.load_data_with_doc(
            args, filename_train_docs)
    logger.info(len(train_docs))
    logger.info(len(train_questions))
    
    train_exs_with_doc = read_data(filename_train, train_questions, train_len)
    logger.info('Num train examples = {}'.format(str(len(train_exs_with_doc))))
    
    dev_docs, dev_questions, _ = utils.load_data_with_doc(
            args, filename_dev_docs)
    logger.info(len(dev_docs))
    logger.info(len(dev_questions))
    
    dev_exs_with_doc = read_data(filename_dev, dev_questions)
    logger.info('Num dev examples = {}'.format(str(len(dev_exs_with_doc))))
    
    test_docs, test_questions, _ = utils.load_data_with_doc(
            args, filename_test_docs)
    logger.info(len(test_docs))
    logger.info(len(test_questions))
    
    test_exs_with_doc = read_data(filename_test, test_questions)
    logger.info('Num test examples = {}'.format(str(len(test_exs_with_doc))))


    # --------------------------------------------------------------------------
    # MODEL SETUP
    # -------------------------------------------------------------------------   
    
    logger.info('-' * 100)
    start_epoch = 0
    
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file)
        start_epoch = 0
    
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained, args)
            
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                
                # Add words in training and dev examples
                #words = utils.load_words(args, train_exs + dev_exs)
                words = utils.load_words(
                        args, train_exs_with_doc + dev_exs_with_doc)
                added = model.expand_dictionary(words)
                
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_docs)

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()
    
    # GPU usage
    if args.show_cuda_stats:
        gpu_usage()

    
    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # -------------------------------------------------------------------------   
    
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    
    # best practices for memory management are available here:
    # https://pytorch.org/docs/stable/notes/cuda.html#best-practices
    
    train_dataset_with_doc = data.ReaderDataset_with_Doc(
            train_exs_with_doc, model, train_docs, single_answer=True)
    train_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(
            train_dataset_with_doc)
    train_loader_with_doc = torch.utils.data.DataLoader(
            train_dataset_with_doc,
            batch_size=args.batch_size, # batch_size of 128 samples
            sampler=train_sampler_with_doc,
            num_workers=args.data_workers, # num_workers increased to 12
            collate_fn=vector.batchify_with_docs,
            pin_memory=args.cuda, # pin_memory = True by default
            )

    dev_dataset_with_doc = data.ReaderDataset_with_Doc(
            dev_exs_with_doc, model, dev_docs, single_answer=False)
    dev_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(
            dev_dataset_with_doc)
    dev_loader_with_doc = torch.utils.data.DataLoader(
            dev_dataset_with_doc,
            batch_size=args.test_batch_size,
            sampler=dev_sampler_with_doc,
            num_workers=args.data_workers,
            collate_fn=vector.batchify_with_docs,
            pin_memory=args.cuda,
            )

    test_dataset_with_doc = data.ReaderDataset_with_Doc(
            test_exs_with_doc, model, test_docs, single_answer=False)
    test_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(
            test_dataset_with_doc)
    test_loader_with_doc = torch.utils.data.DataLoader(
           test_dataset_with_doc,
           batch_size=args.test_batch_size,
           sampler=test_sampler_with_doc,
           num_workers=args.data_workers,
           collate_fn=vector.batchify_with_docs,
           pin_memory=args.cuda,
           )


    # -------------------------------------------------------------------------
    # PRINT CONFIG 
    # -------------------------------------------------------------------------   
    
    logger.info('-' * 100)
    logger.info('CONFIG:')
    print(json.dumps(vars(args), indent=4, sort_keys=True))


    # --------------------------------------------------------------------------
    # TRAIN/VALIDATION LOOP
    # -------------------------------------------------------------------------   
    
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
          
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        logger.info('-' * 100)
        logger.info('Mode: ' + args.mode)
        
        if (args.mode == 'all'):
            train(args, 
                    train_loader_with_doc, model, stats, 
                    train_exs_with_doc, train_docs)
        if (args.mode == 'reader'):
            pretrain_reader(args, 
                    train_loader_with_doc, model, stats, 
                    train_exs_with_doc, train_docs)
        if (args.mode == 'selector'):
            pretrain_selector(args, 
                    train_loader_with_doc, model, stats, 
                    train_exs_with_doc, train_docs)
        
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # -----------------------------------------------------------------
            result = validate_with_doc(args, 
                    dev_loader_with_doc, model, stats, dev_exs_with_doc, 
                    dev_docs, 'dev')
            
            validate_with_doc(args, 
                    train_loader_with_doc, model, stats, train_exs_with_doc, 
                    train_docs, 'train')
            
            if (dataset=='webquestions' or dataset=='CuratedTrec'): # not applicable
                result = validate_with_doc(args, 
                        test_loader_with_doc, model, stats, 
                        test_exs_with_doc, test_docs, 'test')
            else: # dataset == 'searchqa' by default, 'squad', 'quasart' or 'unftriviaqa'
                validate_with_doc(args, 
                        test_loader_with_doc, model, stats, 
                        test_exs_with_doc, test_docs, 'test')
        # ---------------------------------------------------------------------
        
        # Save model with improved evaluation results
        if result[args.valid_metric] > stats['best_valid']:
            
            txt = 'Best valid: {} = {:.2f} (epoch {}, {} updates)'
            logger.info(txt.format(
                    args.valid_metric, result[args.valid_metric],
                    stats['epoch'], model.updates))
            
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]
        
        # Clean the gpu before running a new iteration
        if args.cuda:                 
            
            gc.collect() # force garbage collection
            for obj in gc.get_objects(): 
                if torch.is_tensor(obj): 
                    del obj
            
            torch.cuda.synchronize(device=model.device) # wait for the gpu
            torch.cuda.empty_cache() # force garbage removal
        
        # CUDA memory
        txt_cuda(show=True, txt='after garbage collection')
