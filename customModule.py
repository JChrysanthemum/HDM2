from __future__ import print_function
import argparse
from random import Random
from re import X

from typing import List
import torch
from torch import Tensor
from torch.cuda import Stream
import torch.nn as nn
from torch.nn import Parameter,init
import math
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
import os,sys
import random
from torchvision import datasets, transforms
import time

class streamLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, stream: Stream) -> Tensor:

        # Original code 
        # return F.linear(input, self.weight, self.bias)
        # x, y = input.shape
        # if y != self.in_features:
        #     print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
        #     return 0

        with torch.cuda.stream(stream):
            # input.matmul(input)

            output = input.matmul(self.weight.t())
            # if self.bias is not None:
                # output += self.bias
            # ret = output.clone() 

            return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def stream_relu(x:Tensor,stream):
    with torch.cuda.stream(stream):
        return  x.matmul(x > 0).clone()
    
class streamReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input:Tensor, stream:Stream):
        '''
        Forward pass of the function.
        '''
        return stream_relu(input,stream) # simply apply already implemented

class Net(nn.Module):
    def __init__(self,in_features: int, out_features: int) -> None:
        super().__init__()
        self.n1=nn.Linear(in_features,240)
        self.n2=nn.Linear(240,240)
        self.n3=nn.Linear(240,240)
        self.n4=nn.Linear(240,240)
        self.n5=nn.Linear(240,240)
        self.n6=nn.Linear(240,out_features)
    
    def forward(self, input: Tensor) -> Tensor:
        x=self.n1(input)
        x=self.n2(x)
        x=self.n3(x)
        x=self.n4(x)
        x=self.n5(x)
        x=self.n6(x)

        return x


class stream_Net(nn.Module):
    def __init__(self,in_features: int, out_features: int) -> None:
        super().__init__()
        self.n1=streamLinear(in_features,240)
        self.n2=streamLinear(240,240)
        self.n3=streamLinear(240,240)
        self.n4=streamLinear(240,240)
        self.n5=streamLinear(240,240)
        self.n6=streamLinear(240,out_features)
    
    def forward(self, input: Tensor, stream: Stream) -> Tensor:
        x=self.n1(input,stream)
        x=self.n2(x,stream)
        x=self.n3(x,stream)
        x=self.n4(x,stream)
        x=self.n5(x,stream)
        x=self.n6(x,stream)

        return x

class stream_Net_Dedi(nn.Module):
    def __init__(self,in_features: int, out_features: int) -> None:
        super().__init__()
        self.n1=streamLinear(in_features,240)
        self.n2=streamLinear(240,240)
        self.n3=streamLinear(240,240)
        self.n4=streamLinear(240,240)
        self.n5=streamLinear(240,240)
        self.n6=streamLinear(240,out_features)
    
    def forward(self, input: Tensor, stream: Stream) -> Tensor:
        # print(stream)
        with torch.cuda.stream(stream):
            x= input.matmul(self.n1.weight.t())
            x= x.matmul(self.n2.weight.t())
            x= x.matmul(self.n3.weight.t())
            x= x.matmul(self.n4.weight.t())
            x= x.matmul(self.n5.weight.t())
            x= x.matmul(self.n6.weight.t())
            return x


def para_test(x, device = 'cuda:0'):
    # nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=all -x true --force-overwrite true --gpu-metrics-device=2 -o streamLinear /home/jiangxt21/HDM/venv/bin/python customModule.py 
    
    ss=[torch.cuda.Stream(device=device) for _ in range(len(ns))]

    def run(iters=15):
        # device = torch.device(0)
        
        for i in range(iters):
            torch.cuda.nvtx.range_push('iter{}'.format(i))
    
            for j in range(len(ns)):
                # print(j)
                # with torch.cuda.stream(ss[j]):
                ns[j](x,ss[j])
            
            torch.cuda.nvtx.range_pop()

    run()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.cudart().cudaProfilerStop()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
ss = [torch.cuda.Stream(device='cuda:0') for _ in range(10)]

def train(x,i,models):
    # global ss 
    s=ss[i]
    for model in models:
        model(x,s)


def mp_test(x, device = 'cuda:0'):
    # nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=all -x true --force-overwrite true --gpu-metrics-device=2 -o streamLinear /home/jiangxt21/HDM/venv/bin/python customModule.py 
    img_size=28
    channels = 1
    batch_size=256
    device = 'cuda:0'
    num_run = 10000
    num_processes = 5
    # global ss 
    # ss = [torch.cuda.Stream(device=device) for _ in range(num_processes)]
    ns = [stream_Net_Dedi(img_size*img_size*channels,10).to(device)]*num_run
    ave = num_run // num_processes

    
    mp.set_start_method('spawn')
    for n in ns:
        n.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    
    start = time.time()
    for i in range(num_processes):
        p = mp.Process(target=train, args=(x,i,ns[i*ave:(i+1)*ave]))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    # for p in processes:
    #     p.join()
    # torch.cuda.synchronize()
    end = time.time()
    print("Multi %d" % num_processes,end-start)


def sp_test(x, device = 'cuda:0'):
    # nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=all -x true --force-overwrite true --gpu-metrics-device=2 -o streamLinear /home/jiangxt21/HDM/venv/bin/python customModule.py 
    img_size=28
    channels = 1
    batch_size=256
    device = 'cuda:0'
    num_run = 10000
    ns = [Net(img_size*img_size*channels,10).to(device)]*num_run


    start = time.time()
    for i in range(num_run):
        ns[i](x)
    end = time.time()
    print("Single",end-start)



if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    img_size=28
    channels = 1
    batch_size=10
    device = 'cuda:0'
    
    ## Layer with big tensor
    # ns = [streamLinear(batch_size,batch_size).to(device)]*3
    # x = torch.rand(size=(batch_size,batch_size)).to(device)

    # ns = [stream_Net_Dedi(img_size*img_size*channels,10).to(device)]*10
    x = torch.rand(size=(batch_size,img_size*img_size*channels)).to(device)

    # para_test(ns,x,device)
    # mp_test(x,device)
    sp_test(x,device)




