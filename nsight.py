
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=all -x true -o my_profile --force-overwrite true --gpu-metrics-device=2 /home/jiangxt21/HDM/venv/bin/python nsight.py

# setup
device = 'cuda:0'

def resnet_test():

    model = models.resnet18().to(device)
    data = torch.randn(64, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (64,), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    nb_iters = 20
    warmup_iters = 10
    for i in range(nb_iters):
        optimizer.zero_grad()
    
        # start profiling after 10 warmup iterations
        if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
    
        # push range for current iteration
        if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))
    
        # push range for forward
        if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
        output = model(data)
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    
        loss = criterion(output, target)
    
        if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
        loss.backward()
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    
        if i >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
        optimizer.step()
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    
        # pop iteration range
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    
    torch.cuda.cudart().cudaProfilerStop()

def seq_para():
    t_times=2048
    t_size=1

    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)
    x = torch.rand(size=(t_size*t_times, t_size*t_times)).to(device)
    w1 = torch.rand(size=(t_size*t_times, t_size*t_times)).to(device)
    w2 = torch.rand(size=(t_size*t_times, t_size*t_times)).to(device)

    def run(iters=15):
        # device = torch.device(0)
        
        for i in range(iters):
            torch.cuda.nvtx.range_push('iter{}'.format(i))
    
            with torch.cuda.stream(s1):
                out1 = x.matmul(w1)
        
            with torch.cuda.stream(s2):
                out2 = x.matmul(w2)
                
            torch.cuda.nvtx.range_pop()
            
    # warmup
    run()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.cudart().cudaProfilerStop()

# seq_para()

def layer_para():
    img_size=28
    batch_size=2048


    n1,n2 = nn.Linear(1 * img_size * img_size, 120).to(device),nn.Linear(1 * img_size * img_size, 120).to(device),

    x = torch.rand(size=(batch_size,1,img_size,img_size)).to(device)
    x = torch.flatten(x, 1)
    s1,s2,s3 = torch.cuda.Stream(device=device),torch.cuda.Stream(device=device),torch.cuda.Stream(device=device)

    def run(iters=15):
        # device = torch.device(0)
        
        for i in range(iters):
            torch.cuda.nvtx.range_push('iter{}'.format(i))
    
            with torch.cuda.stream(s1):
                F.relu(n1(x))
    
            with torch.cuda.stream(s2):
                F.relu(n2(x))
            
            torch.cuda.nvtx.range_pop()

    run()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.cudart().cudaProfilerStop()

layer_para()

def model_para():
    from test import Net
    img_size=28
    batch_size=2048


    n1,n2,n3 = Net(img_size).to(device),Net(img_size).to(device),Net(img_size).to(device)
    x = torch.rand(size=(batch_size,1,img_size,img_size)).to(device)
    s1,s2,s3 = torch.cuda.Stream(device=device),torch.cuda.Stream(device=device),torch.cuda.Stream(device=device)

    def run(iters=15):
        # device = torch.device(0)
        
        for i in range(iters):
            torch.cuda.nvtx.range_push('iter{}'.format(i))
    
            with torch.cuda.stream(s1):
                n1(x)
    
            with torch.cuda.stream(s2):
                n2(x)
            
            with torch.cuda.stream(s3):
                n3(x)
                
            torch.cuda.nvtx.range_pop()

    run()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.cudart().cudaProfilerStop()

# model_para()


