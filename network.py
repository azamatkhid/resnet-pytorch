import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,inchannel,outchannel,stride=1,act=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.act1=act(inplace=True)
        self.conv2=nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.act2=act(inplace=True)
        self.skip=nn.Sequential()
        if stride>1 or inchannel!=outchannel:
            self.skip=nn.Sequential(nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,padding=0,bias=False),
                    nn.BatchNorm2d(outchannel))
    
    def forward(self,x):
        out=self.act1(self.bn1(self.conv1(x)))
        out=self.act2(self.bn2(self.conv2(out))+self.skip(x))
        return out

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inchannel,outchannel,stride=1,act=nn.ReLU):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Conv2d(inchannel,outchannel,stride=1,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(outchannel)
        self.act1=act(inplace=True)
        self.conv2=nn.Conv2d(outchannel,outchannel,stride=stride,kernel_size=3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.act2=act(inplace=True)
        self.conv3=nn.Conv2d(outchannel,outchannel*self.expansion,stride=1,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(outchannel*self.expansion)
        self.act3=act(inplace=True)
        self.skip=nn.Sequential()
        if stride>1 or inchannel!=outchannel*self.expansion:
            self.skip=nn.Sequential(nn.Conv2d(inchannel,outchannel*self.expansion,stride=stride,kernel_size=1,bias=False),
                    nn.BatchNorm2d(outchannel*self.expansion))

    def forward(self,x):
        out=self.act1(self.bn1(self.conv1(x)))
        out=self.act2(self.bn2(self.conv2(out)))
        out=self.act3(self.bn3(self.conv3(out))+self.skip(x))
        return out


class ResNet(nn.Module):
    base_channel=64
    def __init__(self,layers=[2,2,2,2],act=nn.ReLU,num_classes=1000,res_block=BasicBlock):
        super(ResNet,self).__init__()
        inchannel,outchannel=3,64
        self._layers=[]

        self.conv0=nn.Conv2d(inchannel,outchannel,kernel_size=7,stride=2,padding=3,bias=False)
        self.b0=nn.BatchNorm2d(outchannel)
        self.act0=act(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_res_layer(res_block,64,layers[0],stride=1,act=act)
        self.layer2=self._make_res_layer(res_block,128,layers[1],stride=2,act=act)
        self.layer3=self._make_res_layer(res_block,256,layers[2],stride=2,act=act)
        self.layer4=self._make_res_layer(res_block,512,layers[3],stride=2,act=act)
        self.pooling=nn.AdaptiveAvgPool2d((1,1))
        self.layer5=nn.Linear(512*res_block.expansion, num_classes)


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

    def _make_res_layer(self,res_block,channel,num_blocks,stride=1,act=nn.ReLU):
        blocks=[]
        inchannel,outchannel=channel,channel
        if stride!=1:
            inchannel=channel//stride*res_block.expansion

        blocks.append(res_block(inchannel,outchannel,stride=stride,act=act)) 
        inchannel=channel*res_block.expansion
        for bl in range(1,num_blocks):
            blocks.append(res_block(inchannel,outchannel,stride=1,act=act))
        return nn.Sequential(*blocks)


if __name__=="__main__":
    resnet18=ResNet(layers=[2,2,2,2],res_block=BasicBlock)
    summary(resnet18,(3,224,224))
    resnet50=ResNet(layers=[3,4,6,3],res_block=Bottleneck)
    summary(resnet50,(3,224,224))
