#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Main OpenQA training and testing script.'''

'''
In order to train and test the model, please follow the following steps:

    #0. cut uncessary gpu memory used by gnome-shell
        reexecute gnome-shell: alt-f2 + r + 'enter' 
        systemctl isolate multi-user
    
    #1. pre-train the paragraph reader: 
        python main.py --batch-size 256 --model-name quasart_reader --num-epochs 5 --dataset quasart --mode reader

    #2. pre-train the paragraph selector: 
        python main.py --batch-size 64 --model-name quasart_selector --num-epochs 5 --dataset quasart --mode selector --pretrained models/quasart_reader.mdl

    #3. train the whole model: 
        python main.py --batch-size 32 --model-name quasart_all --num-epochs 5 --dataset quasart --mode all --pretrained models/quasart_selector.mdl


'''

import os
import sys
import torch

from src import sys_dir

from src.pipeline import model_build
from src.pipeline import model_query

import warnings
#warnings.simplefilter('error') # all warnings are enabled
warnings.simplefilter('ignore') # warnings are ignored
#warnings.simplefilter('default') # left as default (probably silent)

from IPython import get_ipython
ipython = get_ipython()

# ------------------------------------------------------------------------------
# CUDA
# ------------------------------------------------------------------------------

os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.append(sys_dir)

# Using CUDNN backend
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

torch.set_default_dtype(torch.float)


# ------------------------------------------------------------------------------
# USER SELECTION
# ------------------------------------------------------------------------------

'''
train/evaluate/save a model from scratch
    mode == 'reader', 'selector' or 'all'

query an existing model 
    mode == 'query'
'''

user_selection = 1

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    
    if user_selection == 1:

        # Interact with a full model pipeline 
        for mode in ['query']:
            model_query.user_interface(mode)
   
    elif user_selection == 2:
     
        # Train, evaluate and save a reader model only
        for mode in ['reader']:
            model_build.load_train_evaluate_save(mode)

    elif user_selection == 3: 
        
        # Train, evaluate and save a full model pipeline
        for mode in ['reader', 'selector', 'all']:
            
            # restart kernel: ipython.magic("%reset -f")
            
            # run the main routine
            model_build.load_train_evaluate_save(mode)            

    else:
        
        raise RuntimeError(user_selection)
