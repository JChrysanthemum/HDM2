from __future__ import print_function
import argparse
from random import Random
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
import os
import random
from torchvision import datasets, transforms
from test import Net

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device("cuda:0")
img_size=28
batch_size=1

def train(rank, x, model):
    print('%d process start'% rank)

    s = torch.cuda.Stream(device=device)

    # torch calc
    seed = random.randint(12,5000)
    print(seed)
    with torch.cuda.stream(s):
        for i in range(seed):
            x.matmul(x)

    # torch model
    # model(x)

    print('%d process end'% rank)


if __name__ == '__main__':
   
    num_processes = 5
    
    
    mp.set_start_method('spawn')

    model = Net(img_size).to(device)
    x = torch.rand(size=(batch_size,1,img_size,img_size)).to(device)

    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, x,model))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    # for p in processes:
    #     p.join()