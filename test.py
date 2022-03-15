import os
import time
from os.path import join as pj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
import numpy as np
import sys, os
from pathlib import Path
from torch import Tensor
import time

# from pynvml import *
# nvmlInit()
# h = nvmlDeviceGetHandleByIndex(0)
# info = nvmlDeviceGetMemoryInfo(h)
# print(f'total    : {info.total}')
# print(f'free     : {info.free}')
# print(f'used     : {info.used}')
# exit()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


def list_files(_path):
    return [f for f in listdir(_path) if isfile(pj(_path, f))]

def mkdir2(_path,renew=False):
    if os.path.exists(_path):
        if renew:
            shutil.rmtree(_path)
        else:
            return
    os.mkdir(_path)

class Net(nn.Module):
    def __init__(self,img_size):
        super().__init__()
        self.fc1 = nn.Linear(1 * img_size * img_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = Net()

class Combined(nn.Module):
    def __init__(self,herd_size):
        super().__init__()
        # self.n1 = Net()
        # self.n2 = Net()
        # self.n3 = Net()
        # self.nets=[Net() for i in range(herd_size)]
        
        # self.nets = [Net() for i in range(herd_size)]\
        self.herd_size = herd_size
        self.nets = []
        self.optims = []
        for i in range(herd_size):
            net=Net()
            net.cuda()
            optimizer = optim.Adam(net.parameters(), lr = 0.01)
            self.nets.append(net)
            self.optims.append(optimizer)

    def forward(self, x,y):
        
        # return torch.stack([self.n1(x),self.n2(x),self.n3(x)])
        for i in range(self.herd_size):
            net=Net()
            optimizer = optim.Adam(net.parameters(), lr = 0.01)
            self.nets.append(net)
            self.optims.append(optimizer)

        return torch.stack([net(x) for net in self.nets])

def cuda_mem(cuda_id=0):
    t = torch.cuda.get_device_properties(cuda_id).total_memory
    r = torch.cuda.memory_reserved(cuda_id)
    a = torch.cuda.memory_allocated(cuda_id)
    print(t,r,a)

def main():
    
    loss = torch.nn.MSELoss(reduction='none')
    
    # a=Tensor([[1],[2]])
    # b=Tensor([[3],[4]])
    # c=Tensor([1,2,3])
    # d=Tensor([3,4,5])
    
    # print(loss(a,b))
    # print(loss(c,d))
    
    
    
    
    _root_path = Path(os.path.dirname(os.path.realpath(__file__))).absolute()
    _data_path = pj(_root_path,"mnist")
    _root_path = pj(_root_path,"result","mlptest")
    
    mkdir2(_data_path)
    mkdir2(_root_path)
    herd_size=256
    img_size=32
    batch_size=64
    n_epochs=50
    
    train_dataset = torchvision.datasets.MNIST(
        root=_data_path, train=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
        target_transform=torchvision.transforms.Compose([
        lambda x:torch.LongTensor([x]), # or just torch.tensor
        lambda x:F.one_hot(x,10),
        lambda x:torch.squeeze(x,0)
        ]),
        download=True)
        
    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    
    
    
    cuda_id = 0
    torch.cuda.set_device(cuda_id)
    loss_func = nn.CrossEntropyLoss()
    
    
    
    # cuda_mem()
    # cb = Combined()
    # cb.cuda()
    # # cb.train()
    
    # cuda_mem()
    
    time_start = time.time()
    for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                imgs = Variable(imgs)
                print(imgs.size())
                exit()
    
                duplabels = labels.repeat(herd_size,1).reshape(labels.size(0),herd_size,-1)
                duplabels = Variable(duplabels)
    
                # print(net(imgs).size())
                # print(cb(imgs).size())
                # print(labels.size())
                
                # print(labels.squeeze(1).size())
                # exit()
                # labels=labels[1:4]
                # print(labels.size())
                # duplabels = labels.repeat(3,1).reshape(labels.size(0),3,-1)
                out = cb(imgs)
                # print(labels)
                print(loss_func(duplabels,out))
    
                # clear gradients for this training step   
                optimizer.zero_grad()           
    
                # backpropagation, compute gradients 
                loss.backward()                # apply gradients
    
                # print(labels.size(),duplabels.size())
                exit()
    
    print("Training %.3f"%(time.time()-time_start))

if __name__ == "__main__":
    main()


