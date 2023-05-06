# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2022/10/16 10:48

import torch
import numpy as np
import random

def seed_everywhere(SEED):
    torch.manual_seed(SEED)  # Current CPU
    torch.cuda.manual_seed(SEED)  # Current GPU
    torch.cuda.manual_seed_all(SEED)  # All GPU (Optional)

    np.random.seed(SEED)  # Numpy module
    random.seed(SEED)  # Python random module
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False  # Close optimization
    # torch.backends.cudnn.deterministic = True  # Close optimization
    print(f"Setting SEED: [{SEED}]")


    # CHECK DEVICE
    # device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: [{device}] is available")
    print(f"cuda device count: {torch.cuda.device_count()}")
    print(f"cuda name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"curr_gpuid: {torch.cuda.current_device()}")
