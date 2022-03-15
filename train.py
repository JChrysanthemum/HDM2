import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T
from torchvision import datasets
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


def train_norm(model:torch.nn.Module,num_epochs = 100, batch_size=100):
    cuda = True 
    cuda_id=0
    torch.cuda.set_device(cuda_id)
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

    dataloader =  torch.utils.data.DataLoader(train_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=1)
    loss_func = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    if cuda:
        model.cuda()
        loss_func.cuda()

    model.train()
            
    # Train the model
    total_step = len(dataloader)
    correct = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            # gives batch data, normalize x when iterate train_loader
            # print(images.size(),images.T.size())
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = model(b_x)      
            loss = loss_func(output, b_y)

            pred_y = torch.max(output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum()/batch_size
            # print(pred_y, labels, labels.size())
            # exit()

            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                accuracy = correct/100
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), accuracy))
                correct = 0


def val_norm(model:torch.nn.Module,num_epochs = 10, batch_size=100):
    cuda = True 
    cuda_id=0
    torch.cuda.set_device(cuda_id)
    transforms = T.Compose([
        T.ToTensor(), 
        T.Lambda(lambda x: torch.flatten(x)),  # (batch, dims)
        # T.Lambda(lambda x: x.T)  # (dims, batch)
        ])

    test_data = datasets.MNIST(
        root = 'data',
        train = False,                         
        transform=transforms, 
        download = True,            
    )

    dataloader =  torch.utils.data.DataLoader(test_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=1)
    if cuda:
        model.cuda()

    model.eval()
            
    # Train the model
    total_step = len(dataloader)
    correct = 0

    for i, (images, labels) in enumerate(dataloader):
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        # gives batch data, normalize x when iterate train_loader
        # print(images.size(),images.T.size())
        b_x = Variable(images)   # batch x
        b_y = Variable(labels)   # batch y

        output = model(b_x)      

        pred_y = torch.max(output, 1)[1].data.squeeze()
        correct += (pred_y == labels).sum()/batch_size
            
    accuracy = correct/len(dataloader)
    print ('Val Acc: {:.4f}' .format(accuracy))

        
def train(model:torch.nn.Module,num_epochs = 100, batch_size=100, detailed=False):
    cuda = True 
    cuda_id=0
    torch.cuda.set_device(cuda_id)
    if not hasattr(model,"herd_size"):
        raise Exception("Herd size not defined")
    herd_size = model.herd_size

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

    dataloader =  torch.utils.data.DataLoader(train_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=1)
    # loss_func = nn.CrossEntropyLoss() 
    loss_func = nn.CrossEntropyLoss(reduction='none') 

    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    if cuda:
        model.cuda()
        loss_func.cuda()

    model.train()
            
    # Train the model
    total_step = len(dataloader)
    correct = 0

    for epoch in range(num_epochs):
        # modelsize datasize batch
        for i, (images, labels) in enumerate(dataloader):
            images = torch.unsqueeze(images.T,0).repeat(herd_size,1,1)
            labels = torch.unsqueeze(labels.T,0).repeat(herd_size,1)
            
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            # gives batch data, normalize x when iterate train_loader
            # print(images.size(),images.T.size())
            
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = model(b_x)  
            # print(output.size(),b_y.size())
            # exit()    
            loss = loss_func(output, b_y)
            pred_y = torch.max(output, 1)[1].data.squeeze()

            # correct value as a scalar tensor
            # correct += (pred_y == labels).sum()/(batch_size*herd_size)

            # correct value for each model
            correct += (pred_y == labels).sum(1)/batch_size
            # print(pred_y, labels, labels.size())
            # exit()

            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            # loss.backward()                # apply gradients       
            loss.backward(torch.ones_like(loss.data))
            optimizer.step()                
            
            if (i+1) % batch_size == 0:
                correct = correct/batch_size
                print ('Herd {} Epoch [{}/{}], Step [{}/{}]' .format(herd_size,epoch + 1, num_epochs, i + 1, total_step))
                if not detailed:                 
                    c = correct.tolist()
                    l = loss.sum(1).tolist()
                    print("Accuracy: Min [%.4f], Max [%.4f], Avg [%.4f]"%(min(c), max(c), sum(c)/len(c)))
                    print("Loss: Min [%.4f], Max [%.4f], Avg [%.4f]"%(min(l), max(l), sum(l)/len(l)))
                else:
                    for i, (c,l) in enumerate(zip(correct,loss)):
                        print("Sub Model %04d, Accuracy %.4f, Loss %.4f" % (i,c.item(),l.sum().item()))
                    correct = 0
                # exit()


def val(model:torch.nn.Module,num_epochs = 10, batch_size=100):
    cuda = True 
    cuda_id=0
    torch.cuda.set_device(cuda_id)
    transforms = T.Compose([
        T.ToTensor(), 
        T.Lambda(lambda x: torch.flatten(x)),  # (batch, dims)
        # T.Lambda(lambda x: x.T)  # (dims, batch)
        ])

    test_data = datasets.MNIST(
        root = 'data',
        train = False,                         
        transform=transforms, 
        download = True,            
    )

    dataloader =  torch.utils.data.DataLoader(test_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=1)
    if cuda:
        model.cuda()

    model.eval()
            
    # Train the model
    total_step = len(dataloader)
    correct = 0

    for i, (images, labels) in enumerate(dataloader):
        images = torch.unsqueeze(images.T,0).repeat(herd_size,1,1)
        labels = torch.unsqueeze(labels.T,0).repeat(herd_size,1,1)
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        # gives batch data, normalize x when iterate train_loader
        # print(images.size(),images.T.size())
        b_x = Variable(images)   # batch x
        b_y = Variable(labels)   # batch y

        output = model(b_x)      

        pred_y = torch.max(output, 1)[1].data.squeeze()
        correct += (pred_y == labels).sum()/batch_size
            
    accuracy = correct/len(dataloader)
    print ('Val Acc: {:.4f}' .format(accuracy))

def test():
    # model = nn.Linear(1*28*28,10)

    # model = nn.Sequential(
    #     nn.Linear(1*28*28,10),
    # )

    from combined import myLinear
    model = myLinear(1*28*28,10)

    val_norm(model)
    train_norm(model)
    val_norm(model)

def herd():
    from model import TestHerd
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # for herd_size in [16,32,64,128,256]:

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    for herd_size in [1024,2048]:
        # herd_size = 2048
        net=TestHerd(herd_size=herd_size)
        train(net,detailed = False)
        torch.save(net,"%04d.pt"%herd_size)
    

if __name__ == "__main__":
    herd()