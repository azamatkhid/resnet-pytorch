import os
import torch
from torchsummary import summary

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from network import ResNet
from constants import default

class Model:
    def __init__(self,**kwargs):
        default.update(kwargs)

        self.layers=default["layers"]
        self.epochs=default["epochs"]
        self.batch_size=default["batch_size"]
        self.log_dir=default["log_dir"]
        self.ckpts_dir=default["ckpts_dir"]
        self.lr=default["lr"]
        self.momentum=default["momentum"]
        self.verbose_step=default["verbose_step"]

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.train_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        self.test_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


        self.net=ResNet(self.layers)
        if torch.cuda.device_count()>0:
            self.net=nn.DataParallel(self.net)
            print(f"Number of GPUs {torch.cuda.device_count()}")
        self.net.to(self.device)

    def net_summary(self):
        summary(self.net,(3,224,224))

    def train(self, criterion=nn.CrossEntropyLoss,optimizer=torch.optim.SGD):
        self._check_dirs()
        self._load_data("train")

        writer=SummaryWriter(log_dir=self.log_dir)
        self.criterion=criterion()
        self.optimizer=optimizer(self.net.parameters(),lr=self.lr,momentum=self.momentum)


        iteration=1
        for epch in range(self.epochs):
            running_loss=0.0
            for idx, batch in enumerate(self.train_data,start=0):
                inputs,labels=batch[0].to(self.device),batch[1].to(self.device)
                self.optimizer.zero_grad()
                outputs=self.net(inputs)
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
                if idx%self.verbose_step==self.verbose_step-1:
                    writer.add_scalar("Loss/Train",running_loss/self.verbose_step/self.batch_size,iteration)
                    iteration+=1
                    running_loss=0.0
        torch.save(self.net.state_dict(),self.ckpts_dir)

    
    def test(self):
        pass

    def _check_dirs(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.ckpts_dir):
            os.mkdir(self.ckpts_dir)

    def _load_data(self,*args):
        if args[0]=="train":
            data=datasets.CIFAR10(root="./data",train=True,download=True,transform=self.train_transforms)
            self.train_data=torch.utils.data.DataLoader(data, batch_size=self.batch_size,shuffle=True,num_workers=1)
        elif args[0]=="test":
            data=datasets.CIFAR10(root="./data",train=True,download=True,transform=self.train_transforms)
            self.test_data=torch.utils.data.DataLoader(data, batch_size=self.batch_size,shuffle=False,num_workers=1)




