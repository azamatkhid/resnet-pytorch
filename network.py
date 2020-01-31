import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class ResBlock2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=0,stride=1,activation=nn.ReLU):
        super(ResBlock2,self).__init__()
        self.res_block=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True),
                nn.Conv2d(out_channels,out_channels,kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True))
        self.skip_conv=nn.Conv2d(in_channels,out_channels,1)

    def forward(self,x):
        residual=x
        x=self.res_block(x)
        if x.size()[1]!=residual.size()[1]:
            residual=self.skip_conv(residual)
        return x+residual

class ResBlock3(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=0,stride=1,activation=nn.ReLU):
        super(ResBlock3,self).__init__()
        self.res_block=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True),
                nn.Conv2d(out_channels,out_channels,kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True),
                nn.Conv2d(out_channels,out_channels,kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True))
        self.skip_conv = nn.Conv2d(in_channels,out_channels,1)
        
    def forward(self,x):
        residual=x
        x=self.res_block(x)
        if x.size()[1]!=residual.size()[1]:
            residual=self.skip_conv(residual)
        return x+residual

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0, pool=False):
        super(BasicBlock,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
        self.pool=pool
        if pool:
            self.pool=nn.MaxPool2d(kernel_size=2)
    
    def forward(self,x):
        x=self.conv(x)
        if self.pool:
            return self.pool(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_blocks=1):
        super(ResNet,self).__init__()
        blocks=[]
        in_channels, out_channels=3,64
        blocks.append(BasicBlock(in_channels,out_channels,7,pool=True))
        in_channels=out_channels
        for i in range(num_blocks):
            blocks.append(ResBlock2(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1))

        self.net=nn.Sequential(*blocks)

    def forward(self,x):
        return self.net(x)

if __name__=="__main__":
    net=ResNet(num_blocks=2)
    print(summary(net,(3,224,224)))

    model=models.resnet18()
    print(summary(model,(3,224,224)))
