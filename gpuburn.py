import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch

x = torch.linspace(0, 4, 128*1024**2).cuda()

while True:
    x = x * (1.0 - x)
