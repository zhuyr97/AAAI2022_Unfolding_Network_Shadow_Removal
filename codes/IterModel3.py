import torch
import torch.nn as nn
import torch.nn.functional as F
from IterModel3_model import A_net,NBNetUnet_initA,A_net2
#from thop import profile

class Deshadow_netS4(nn.Module):
    def __init__(self,ex1=6,ex2=4):
        super(Deshadow_netS4, self).__init__()
        self.init_net = NBNetUnet_initA(expansion=ex1)
        self.iters_A_net = A_net2(expansion=ex2)
    def forward(self, inputs,mask):
        listA =[]
        listJ =[]
        #init step
        A0,J0 = self.init_net(inputs,mask)
        J0 = (1 + A0)*inputs
        listA.append(A0)
        listJ.append(J0)
        #iter1
        A1 = self.iters_A_net(inputs,J0,A0,mask)
        J1 = (1 + A1) * inputs
        listA.append(A1)
        listJ.append(J1)
        # iter2
        A2 = self.iters_A_net(inputs, J1, A1, mask)
        J2 = (1 + A2) * inputs
        listA.append(A2)
        listJ.append(J2)
        # iter3
        A3 = self.iters_A_net(inputs, J2, A2, mask)
        J3 = (1 + A3) * inputs
        listA.append(A3)
        listJ.append(J3)
        # iter4
        A4 = self.iters_A_net(inputs, J3, A3, mask)
        J4 = (1 + A4) * inputs
        listA.append(A4)
        listJ.append(J4)
        return listA,listA[-1],listJ,listJ[-1]

if __name__ == "__main__":
    model = Deshadow_netS4()
    input = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    _,weight_map,_,output = model(input,mask)
    print('-'*50)
    print(output.shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    #flops, params = profile(model, inputs=(input,mask,))
    #print(flops, params, '----', flops / 1000000000, params / 1000000)

