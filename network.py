import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,kernel_size=3,act=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(inchannel,outchannel,kernel_size=kernel_size,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.act1=act(inplace=True)
        self.conv2=nn.Conv2d(outchannel,outchannel,kernel_size=kernel_size,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.act2=act(inplace=True)
        self.skip=nn.Sequential()
        if stride>1 or inchannel!=outchannel:
            self.skip=nn.Sequential(nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,padding=0,bias=False),
                    nn.BatchNorm2d(outchannel))
    
    def forward(self,x):
        out=self.act1(self.bn1(self.conv1(x)))
        out=self.act2(self.bn2(self.conv2(out)))
        out=out+self.skip(x)
        return out

class Bottleneck(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,kernel_size=3,act=nn.ReLU):
        super(Bottleneck,self).__init__()













class ResNet(nn.Module):
    def __init__(self,layers=[2,2,2,2],act=nn.ReLU, num_classes=1000,res_block=BasicBlock):
        super(ResNet,self).__init__()
        inchannel,outchannel=3,64
        self._layers=[]

        self.conv0=nn.Conv2d(inchannel,outchannel,kernel_size=7,stride=2,padding=3,bias=False)
        self.b0=nn.BatchNorm2d(outchannel)
        self.act0=act(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_res_layer(res_block,outchannel,outchannel,layers[0],stride=1,act=act)
        self.layer2=self._make_res_layer(res_block,outchannel,2*outchannel,layers[1],stride=2,act=act)
        self.layer3=self._make_res_layer(res_block,2*outchannel,4*outchannel,layers[2],stride=2,act=act)
        self.layer4=self._make_res_layer(res_block,4*outchannel,8*outchannel,layers[3],stride=2,act=act)
        self.pooling=nn.AdaptiveAvgPool2d((1,1))
        self.layer5=nn.Linear(512, num_classes)


    def forward(self,x):
        out=self.maxpool(self.act0(self.b0(self.conv0(x))))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.pooling(out)
        out=torch.flatten(out,1)
        out=self.layer5(out)
        return out

    def _make_res_layer(self,res_block,inchannel,outchannel,num_blocks,stride=1,act=nn.ReLU):
        blocks=[]
        blocks.append(res_block(inchannel,outchannel,stride=stride,act=act))
        for bl in range(1,num_blocks):
            blocks.append(res_block(outchannel,outchannel,stride=1,act=act))
        return nn.Sequential(*blocks)


if __name__=="__main__":
    net=ResNet(layers=[2,2,2,2])
    summary(net,(3,224,224))
