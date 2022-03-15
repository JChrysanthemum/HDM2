import enum
from torch.nn import Module,Parameter,Sequential,Linear
from torch.autograd import Variable
import torch
import numpy as np
import random
import math
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def display_param(model:Module):
    for name, param in model.named_parameters():
        print(name,': ',param.requires_grad, param.data.size())

class OneNet(Module):
    """Module as a placeholder that doing nothing
    
    """
    def __init__(self,in_features) -> None:
        super().__init__()
        ones = torch.ones(1)
        self.weight = Parameter(ones, requires_grad=False)
        self.register_parameter(name='weight', param=self.weight)
    def forward(self, x):
        return x.matmul(1)

class TestHerd(Module):
    """
    First version of Herd
    # @TODO randomize layer size for each model

    """
    def __init__(self,herd_size=32, train_size=200,input_size=28*28*1,output_size=10) -> None:
        super().__init__()
        self.herd_size = herd_size

        
        # step 1. Calcuate the models height
        # Say we have paralled 32 models, which ranged from 2 layers to 6.
        # That they have different height of parameters.
        
        # model_size_list=[1]*herd_size
        self._ms_min = 2
        self._ms_max = 6
        model_size_list=[random.randint(self._ms_min,self._ms_max) for i in range(herd_size)]
        model_size_list.sort()
        # print(model_size_list)

        height_count = []
        for i in range(1,max(model_size_list)+1): # start at 2 in Herd
            height_count.append(model_size_list.count(i))
        # height_count=height_count[::-1]

        models_height=[]
        for i in range(len(height_count)):
            models_height.append(sum(height_count[i:]))
        
        self.models_height = models_height
        self.herd_layer_max = max(model_size_list)

        # print(self.herd_layer_max,height_count,models_height)

        # example 
        # layer         1   2   3   4   5   6
        # height_count [0,  6,  6,  9,  4,  7]
        # models_height [32, 32, 26, 20, 11, 7]
        # colum 1 has 32 set of params


        # step 2. Calcuate the models layer size
        # Each model has a radomized layer size, but same input and 
        # output size.
        # This will be used in weight initialization.

        # layer size base
        ls = int(np.log(train_size)*26)
        if (ls<120):
            ls=120
        # @TODO randomize layer size for each model
        # layer_size_list = [math.floor(random.gauss(ls,ls//16)) for i in range(herd_size)]
        layer_size_list = [120]*herd_size
        
        self.params = []
        
        # step 2.1 Create input layer and output layer
        input_layer = {
            "name":"input",
            "weight":[],
            "bias":[]
        }
        output_layer = {
            "name":"output",
            "weight":[],
            "bias":[]
        }

        # Weight Param: out_features, in_features
        # Bias Param: out_features
        for _s in layer_size_list:
            input_layer["weight"].append(torch.rand(_s, input_size))
            output_layer["weight"].append(torch.rand(output_size, _s))
            input_layer["bias"].append(torch.rand(_s))
            output_layer["bias"].append(torch.rand(output_size))

        layers = [input_layer,output_layer]

        # step 2.1 Create midle layer
        for i,_h in enumerate(models_height):
            # print(len(layer_size_list[:_h]))
            _layer = {
                "name":'%04d'%(i+1),
                "weight":[],
                "bias":[]
            }
            for _s in layer_size_list[:_h]:
                _layer["weight"].append(torch.rand(_s, _s))
                _layer["bias"].append(torch.rand(_s))
            layers.insert(i+1,_layer)
 
        self.combined_weight = []
        self.combined_bias = []
        # Combined Weight Param: model_size, out_features, in_features
        # Combined Bias Param: model_size, out_features
        for l in layers:
            
            weight = torch.nn.Parameter(torch.stack(l["weight"],dim=0), requires_grad=True)
            bias = torch.nn.Parameter(torch.stack(l["bias"],dim=0), requires_grad=True)

            self.combined_weight.append(weight)
            self.combined_bias.append(bias)
            self.register_parameter(name="%s-weight"%l["name"], param=weight)
            self.register_parameter(name="%s-bias"%l["name"], param=bias)

            # self.combined_weight.append(l["name"])
            # print(l["name"])
            # for s in l["weight"]:
            #     print(s.size())
            # print(weight.size())
            # exit()

        # add Input and Output to models_height
        # print(self.models_height)
        self.models_height.insert(0,herd_size)
        # print(self.models_height)
        self.models_height.append(models_height[-1])
        # print(self.models_height)
        # exit()


        # print(self.combined_weight)
        # print(layers)


        # model_list=[]

        # for ms in model_size_list:
        #     rndls=math.floor(random.gauss(ls,ls//16))
        #     seq = [Linear(rndls,rndls)]*ms
        #     model = Sequential(
        #         Linear(input_size,rndls),
        #         *seq,
        #         Linear(rndls,output_size),
        #     )
        #     model_list.append(model)

        # print(model_size_list)
        # print(model_list)

    def forward(self,x):
        batch_size = x.size(2)
        outs = []

        for i, (weight,bias) in enumerate(zip(self.combined_weight,self.combined_bias)):
            cur_model_size = self.models_height[i]
            bias = torch.unsqueeze(bias,2).repeat(1,1,batch_size)

            # print(x.size())

            if i == self._ms_max+1: # last layer
                # print("Last layer")
                out_size = cur_model_size
                outs.append(x)
                # for o in outs:
                #     print(o.size())
                # print(torch.cat(outs).size())
                x = torch.cat(outs)
                out = weight.matmul(x) + bias
                return out
                
            else:
                out_size =  cur_model_size - self.models_height[i+1]
                # print(i,weight.size(),bias.size(),cur_model_size,out_size)
                x = weight.matmul(x) + bias

                offset = cur_model_size-out_size
                _out = x[cur_model_size-out_size: ,:,:]

                # print("offset",offset)
                if _out.size(0) != 0:
                    outs.append(_out)
                
                x = x[0:offset,:,:]
                # print(cur_model_size-out_size,_out.size())
                
                # print(i,weight.size(),x.size())
            
            


            


        pass

class Herd(Module):
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
            model = Sequential(
                Linear(input_size,rndls),
                *seq,
                Linear(rndls,output_size),
            )
            model_list.append(model)

        print(model_size_list)
        print(model_list)



    def forward(self,x):
        pass


def test():
    herd_size = 32
    input_size = 1*28*28
    output_size = 10
    batch_size = 2

    x=torch.rand(batch_size,input_size)
    # label here should be onehot
    y=torch.rand(batch_size,output_size)

    # modelsize datasize batch
    xs=torch.stack([x.T]*herd_size)
    ys=torch.stack([y.T]*herd_size)
    # ys=torch.stack([y]*model_size)

    xs = Variable(xs)
    ys = Variable(ys)

    net = TestHerd(herd_size=herd_size,input_size=input_size,output_size=output_size)
    # display_param(net)
    net(xs)

if __name__ == "__main__":
    test()

# a=torch.rand(1,2)
# b=torch.rand(3,4)
# print(a,b)
# c=torch.stack([a,b])

