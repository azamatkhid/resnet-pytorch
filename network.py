import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=0,stride=1,activation=nn.ReLU):
        super(ResBlock,self).__init__()
        if stride > 1:
            self.res_block=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride=2,padding=padding,bias=False),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True),
                nn.Conv2d(out_channels,out_channels,kernel_size,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels,out_channels,kernel_size,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True))
        else:
            self.res_block=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True),
                nn.Conv2d(out_channels,out_channels,kernel_size,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True))
 
        self.skip_conv1x1=nn.Conv2d(in_channels,out_channels,1,bias=False)
        self.skip_conv3x3=nn.Conv2d(in_channels,out_channels,kernel_size,stride=2,padding=1,bias=False)

    def forward(self,x):
        residual=x
        x=self.res_block(x)
        if x.size()[1]!=residual.size()[1]:
            if x.size()[2]!=residual.size()[2]:
                residual=self.skip_conv3x3(residual)
            else:
                residual=self.skip_conv1x1(residual)
        x=x+residual
        print(x.size())
        return x

class ResNet(nn.Module):
    def __init__(self, num_blocks=1,activation=nn.ReLU):
        super(ResNet,self).__init__()
        blocks=[]
        in_channels,out_channels=3,64
        kernel_size,stride,padding=7,2,3
        blocks.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False))
        blocks.append(nn.BatchNorm2d(out_channels))
        blocks.append(activation(inplace=True))

        in_channels,out_channels=out_channels,out_channels
        blocks.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        for bl in range(2):
            for i in range(2):
                stride, padding=1,1
                if i==0 and bl>=1:
                    stride=2
                blocks.append(ResBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=padding,stride=stride))
                in_channels=out_channels
            out_channels*=2

        self.net=nn.Sequential(*blocks)

    def forward(self,x):
        return self.net(x)

if __name__=="__main__":
    net=ResNet(num_blocks=2)
    print(summary(net,(3,224,224)))

    model=models.resnet18()
    print(summary(model,(3,224,224)))
