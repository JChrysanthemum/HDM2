import os
from statistics import mode
import time
from os.path import join as pj
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.cuda import Stream
from torch.nn import Parameter,init
import math
import torch.multiprocessing as mp
import torch.optim as optim
import shutil
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
import torchvision.transforms as T
from torchvision.utils import save_image
import torchvision
import numpy as np
import sys, os
from pathlib import Path
from torch import LongTensor, Tensor
import time
import numpy as np
import math,random
import copy

input_size = 100
output_size = 10
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OneNet(nn.Module):
    """Module as a placeholder that doing nothing
    
    """
    def __init__(self,in_features) -> None:
        super().__init__()
        ones = torch.FloatTensor(in_features).fill_(1)
        param = torch.nn.Parameter(ones, requires_grad=False)
        self.register_parameter(name='bias', param=param)
    def forward(self, x):
        return x

class Herd(nn.Module):
    def __init__(self,herd_size=32, train_size=200,input_size=28*28*1,output_size=10) -> None:
        super().__init__()

        # layer size base
        ls = int(np.log(train_size)*26)
        if (ls<120):
            ls=120

        model_size_list=[random.randint(2,6) for i in range(herd_size)]
        model_size_list.sort()

        model_list=[]

        for ms in model_size_list:
            rndls=math.floor(random.gauss(ls,ls//16))
            seq = [Linear(rndls,rndls)]*ms
            model = nn.Sequential(
                Linear(input_size,rndls),
                *seq,
                Linear(rndls,output_size),
            )
            model_list.append(model)
            print(model)

        # print(np.log(train_size),ls,rndls)

        # nls = random.randint(2,6)


    def forward(self,x):
        pass

class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = input.matmul(self.weight.t())
        
        if self.bias is not None:
            output += self.bias
        ret = output
        # print(ret.size())
        # exit()
        return ret
        
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# modelsize datasize batch
class combinedLinear(nn.Module):
    """Combine multiple Linear
                     
    """
    def __init__(self,model_size, in_features,out_features) -> None:

        super().__init__()
        self.model_size = model_size

        # data = torch.FloatTensor(np.random.choice([0,1],(model_size, out_features, in_features)))
        data_weight = torch.rand(model_size, out_features, in_features)
        data_bias = torch.rand(model_size, out_features)

        self.weight = torch.nn.Parameter(data_weight, requires_grad=True)
        self.bias = torch.nn.Parameter(data_bias, requires_grad=True)
        self.register_parameter(name='weight', param=self.weight)
        self.register_parameter(name='bias', param=self.bias)


    def forward(self, x):
        # modelsize datasize batch
        batch_size = x.size(2)
        bias = torch.unsqueeze(self.bias,2).repeat(1,1,batch_size)
        # print(self.bias.size(),self.weight.matmul(x).size(),torch.unsqueeze(self.bias,2).size(),torch.unsqueeze(self.bias,2).repeat(1,1,batch_size).size())
        # exit()
        return self.weight.matmul(x)  +  bias #torch.stack([self.bias.T]*batch_size).T


def display_param(model:nn.Module):
    for name, param in model.named_parameters():
        print(name,': ',param.requires_grad,param.data, param.data.size())

def test():
    class TestNet(nn.Module):
        """Module as a placeholder that doing nothing
                     
        """
        def __init__(self,model_size, in_features,out_features) -> None:

            super().__init__()

            rnd = torch.FloatTensor(np.random.choice([0,1],(model_size, out_features, in_features)))

            param = torch.nn.Parameter(rnd, requires_grad=False)
            self.register_parameter(name='weight', param=param)
        def forward(self, x):
            weight = list(net.parameters())[0]
            # print(x.size(),weight.size())
            return weight.matmul(x)
            # return x.matmul(weight.T)

    
    
    modelsize = 2
    batch_size = 5
    data_size = 3
    label_size = 2
    net = TestNet(model_size=modelsize, in_features=data_size, out_features=label_size)

    x=torch.rand((batch_size,data_size)) 
    # x=torch.ones((batch_size,data_size))
    x=torch.stack([x.T]*modelsize)
    # modelsize datasize batch
    # print(x.size())
    # return 

    print("X: \n", x)
    print("Weight: \n", list(net.parameters())[0])

    print('Result: \n',net(x))
    print('Readable Result: \n',net(x).permute(0,2,1))
    
    # print(net(x)[0,:,0])

    # print(list(net.parameters())[0])
    # print(net(x).size())

def test2():
    # net = Linear(3,2) 
    # net = Net()
    net = nn.Sequential(
        Linear(10,32),
        Linear(32,32),
        Linear(32,10),
    )
    # print(list(net.parameters()))
    # for p in net.parameters():
        # print(p)
    for name, param in net.named_parameters():
        print(name,': ',param.requires_grad,param.data.size())
    # x=torch.rand((256,3))
    # net(x)   

def test_weight_loss():
    class myLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            
        def reset_parameters(self):
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
            
        def forward(self, input):
            x, y = input.shape
            if y != self.in_features:
                print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
                return 0
            output = input.matmul(self.weight.t())
            
            if self.bias is not None:
                output += self.bias
            ret = output
            # print(ret.size())
            # exit()
            return ret
            
        
        def extra_repr(self):
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )
    device = torch.device('cuda:0')

    from torchvision import datasets
    from torchvision.transforms import ToTensor

    transforms = T.Compose([
        T.ToTensor(), 
        T.Lambda(lambda x: torch.flatten(x)),  # (batch, dims)
        # T.Lambda(lambda x: x.T)  # (dims, batch)
        ])

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform=transforms, 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform=transforms,
        download = True
    )
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                              batch_size=100, 
                                              shuffle=True, 
                                              num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(test_data, 
                                              batch_size=100, 
                                              shuffle=True, 
                                              num_workers=1),
    }
    model = myLinear(28*28*1,10)
    # model = Linear(28*28*1,10)
    loss_func = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    num_epochs = 10
    def train(num_epochs, cnn, loaders):
    
        cnn.train()
            
        # Train the model
        total_step = len(loaders['train'])
            
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                
                # gives batch data, normalize x when iterate train_loader
                # print(images.size(),images.T.size())
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = cnn(b_x)      
                # print(output.size())
                loss = loss_func(output, b_y)
                # print(loss)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()                # apply gradients             
                optimizer.step()                
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
                       
    train(num_epochs,model,loaders)

def test_combined_Linear():

    # model_size = 3
    # input_shape = 28*28*1
    # output_shape = 10
    # batch_size = 2

    model_size = 3
    input_shape = 1
    output_shape = 2
    batch_size = 2

    loss_func = nn.CrossEntropyLoss(reduction='none') 
    
    x=torch.rand(batch_size,input_shape)

    # label here should be onehot
    y=torch.rand(batch_size,output_shape)
    # y=torch.Tensor(batch_size)

    # print(y)
    # exit()

    # modelsize datasize batch
    xs=torch.stack([x.T]*model_size)
    ys=torch.stack([y.T]*model_size)
    # ys=torch.stack([y]*model_size)

    xs = Variable(xs)
    ys = Variable(ys)

    net = combinedLinear(model_size,input_shape,output_shape)  
    # net.eval()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr = 0.01)

    for i in range(3):
        output=net(xs)
    
        # print(output.size(), ys.size())
        # print(output, ys)
        print("Diff label", output-ys)
    
        # net.zero_grad()  
        
        loss = loss_func(output, ys)
        # loss += 100
    
        print("Loss: ", loss) 
        # print(loss.requires_grad)
                 
        # print("Weight: ", list(net.parameters())[0]) 
        display_param(net)
        # loss.backward(torch.ones_like(loss.data))
        old_w = copy.deepcopy(net.weight)
        old_b = copy.deepcopy(net.bias)
        # loss.backward()
        loss.backward(torch.ones_like(loss.data))
        print('gradients =', [x.grad.data  for x in net.parameters()] )
        optimizer.step() 
        display_param(net)
        print("Weight Diff:", old_w.data - net.weight.data)
        print("Weight Diff:", old_b.data - net.bias.data)

def test_loss():
    a=Tensor([[0,0,1]])
    b=LongTensor([2])
    c=Tensor([[0,0,1]])

    ls = nn.CrossEntropyLoss()
    print(ls(a,b))
    print(ls(a,c))

    d = torch.rand(1,2,1)
    e = torch.zeros(1,2,1)
    print(d,e, 'DE', ls(d,e))


def temp():
    net = Linear(2,3)
    criterion = nn.CrossEntropyLoss()
    net.train()

    # print(out)
    # print(b)

    x = torch.randn(1, 2)
    target = torch.randint(0, 2, (1,))
    criterion = nn.CrossEntropyLoss()
    
    
    output = net(x)

    print(output,target)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    print(loss)
    # print(model.lstm.weight_ih_l0.grad.abs().sum())
    
    # loss = cri(out,b)
    # print(input, target)
    # print(output)

if __name__ == "__main__":
    # net = Net()
    # net = OneNet(100)
    # print(len(list(net.parameters())))
    # print(list(net.parameters()))
    # layers = [module for module in list(net.modules()) if not isinstance(module, Net)]
    # print(layers)
    # print(list(Linear(3,2).parameters())[0].size())

    # test()

    # test2()

    # test_weight_loss()

    # test_loss()

    test_combined_Linear()

    # temp()

    # a=Tensor([[1,2]])
    # b=torch.stack([a]*1)
    # print(a.size(),b.size())

    # for i in range(10):
    #     n=Herd(10)

    # Linear : Weight*x + bias
