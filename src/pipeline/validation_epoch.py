#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''Main OpenQA validation script.'''

import logging

from src import num_docs, display_num
from src.reader import utils
from src.pipeline import has_answer

# Logger
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Validation loops. Includes both 'unofficial' and 'official' functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------

def validate_with_doc(args, data_loader, model, global_stats, 
                                 exs_with_doc, docs_by_question, mode):
    '''Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    '''
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    logger.info('validate_with_doc')
    
    # Intialize counters
    examples = 0
    
    aa = [0.0 for i in range(num_docs)] # increment only if example has answer
    bb = [0.0 for i in range(num_docs)] # increment regardless
    
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, ex_id = ex[0].size(0), ex[-1]
        
        # ---------------------------------------------------------------------
        # Document Selector
        # ---------------------------------------------------------------------
        
        '''
        ex_with_doc = 
        [tensor]  x1 = document word indices            [batch * len_d]
        [tensor]  x1_f = document word features indices [batch * len_d * nfeat]
        [tensor]  x1_mask = document padding mask       [batch * len_d]
        [tensor]  x2 = question word indices            [batch * len_q]
        [tensor]  x2_mask = question padding mask       [batch * len_q]
        [list]    indices                               [batch]
        '''
        
        scores_doc_num = model.predict_with_doc(ex_with_doc)
        scores = [{} for i in range(batch_size)]
        
        # ---------------------------------------------------------------------
        # Document Reader
        # ---------------------------------------------------------------------
        for idx_doc in range(0, num_docs):
            ex = ex_with_doc[idx_doc]
            pred_s, pred_e, pred_score = model.predict(
                    ex, top_n=display_num)
            
            for i in range(batch_size):
                idx_doc_i = idx_doc %len(docs_by_question[ex_id[i]])
                doc_text = docs_by_question[ex_id[i]][idx_doc_i]['document']
                
                # try to read the 10 best predicted answers (this may trigger 
                # an 'index out of range' exception)
                for k in range(display_num):
                    
                    try:
                        prediction = [doc_text[j] for j in range(pred_s[i][k], 
                                      pred_e[i][k]+1)]
                        prediction = ' '.join(prediction).lower()
                        
                        # update prediction scores
                        if (prediction not in scores[i]): 
                            scores[i][prediction] = 0
                        scores[i][prediction] += (pred_score[i][k] * 
                              scores_doc_num[i][idx_doc])
                    
                    except:
                        pass 
        
        # Get the 10 most likely answers from the batch and see if the answer 
        # is actually in there           
        for i in range(batch_size):
            _, indices = scores_doc_num[i].sort(0, descending = True)
            
            for j in range(0, display_num):
                idx_doc = indices[j]
                idx_doc_i = idx_doc %len(docs_by_question[ex_id[i]])
                
                doc_text = docs_by_question[ex_id[i]][idx_doc_i]['document']
                ex_answer = exs_with_doc[ex_id[i]]['answer']
                
                # Looking for the answer in the document...
                if (has_answer(args, 
                               ex_answer, 
                               doc_text)[0]):
                    aa[j]= aa[j] + 1
                    
                bb[j]= bb[j]+1

        # Update performance metrics
        for i in range(batch_size):
            
            best_score = 0
            prediction = ''
            for key in scores[i]:
                if (scores[i][key] > best_score):
                    best_score = scores[i][key]
                    prediction = key
            
            ground_truths = []
            ex_answer = exs_with_doc[ex_id[i]]['answer']
            
            # Ground truth answers
            if (args.dataset == 'CuratedTrec'): # not applicable
                ground_truths = ex_answer
            else: 
                for a in ex_answer:
                    ground_truths.append(' '.join([w for w in a]))
                    
            exact_match.update(
                    utils.metric_max_over_ground_truths(
                            utils.exact_match_score, prediction, ground_truths))
            
            f1.update(
                    utils.metric_max_over_ground_truths(
                            utils.f1_score, prediction, ground_truths))
            
        examples += batch_size
        
        if (mode=='train' and examples>=1000):
            break
    
    try:
        for j in range(display_num):
            if (j>0):
                aa[j]= aa[j]+aa[j-1]
                bb[j]= bb[j]+bb[j-1]
    except:
        pass
    
    txt =  '{} valid official with doc: Epoch = {} | EM = {:.2f} | '
    txt += 'F1 = {:.2f} | examples = {} | valid time = {:.2f} (s)'
    logger.info(txt.format(
            mode, global_stats['epoch'], exact_match.avg * 100, 
            f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}
