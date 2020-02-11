import os
import torch
import numpy as np

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

from network import ResNet
from constants import default
from tqdm import tqdm

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
        self.verbose=default["verbose"]
        self.mode=default["mode"]

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.train_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        self.test_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        self.net=ResNet(self.layers)
        
        if self.verbose==1:
            summary(self.net,(3,224,224))
        if torch.cuda.device_count()>0:
            self.net=nn.DataParallel(self.net)
            print(f"Number of GPUs {torch.cuda.device_count()}")
        self.net.to(self.device)

    def train(self, criterion=nn.CrossEntropyLoss,optimizer=torch.optim.SGD):
        self.net.train() # required due to BN layer
        self._check_dirs()
        self._load_data("train")

        self.writer=SummaryWriter(log_dir=self.log_dir)
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
                    valid_acc, valid_loss=self._validation()
                    self.writer.add_scalar("Loss/Train",running_loss/self.verbose_step,iteration)
                    self.writer.add_scalar("Loss/Validation",valid_loss,iteration)
                    self.writer.add_scalar("Acc/Validation",valid_acc,iteration)
                    running_loss=0.0
                    iteration+=1

        torch.save(self.net.state_dict(),self.ckpts_dir)

    def _validation(self):
        self.net.eval() # required due to BN layer
        correct=0
        total=0
        total_loss=0
        with torch.no_grad():
            for data in tqdm(self.valid_data):
                inputs,labels=data[0].to(self.device),data[1].to(self.device)
                outputs=self.net(inputs)
                loss=self.criterion(outputs,labels)
                total_loss+=loss.item()
                _,predicted=torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
        self.net.train() # required due BN layer
        acc=correct/total
        batch_loss=total_loss/len(self.valid_data)
        return acc, batch_loss 
    
    def test(self):
        self.net.eval() # required due to BN layer
        self._load_data("test")
        correct=0
        total=0

        with torch.no_grad():
            for data in tqdm(self.test_data):
                inputs,labels=data[0].to(self.device),data[1].to(self.device)
                outputs=self.net(inputs)
                _,predicted=torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
        print(f"Test accuracy: {100*correct/total}%")
        self.net.train() # required due to BN layer

    def _check_dirs(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.ckpts_dir):
            os.mkdir(self.ckpts_dir)

    def _load_data(self,*args):
        if args[0]=="train":
            data=datasets.CIFAR10(root="./data",train=True,download=True,transform=self.train_transforms)

            data_size=len(data)
            indices=list(range(data_size))
            valid_ratio=0.1
            split=int(np.floor(valid_ratio*data_size))
            train_idx,valid_idx=indices[split:],indices[:split]
            train_sampler=SubsetRandomSampler(train_idx)
            valid_sampler=SubsetRandomSampler(valid_idx)

            self.train_data=torch.utils.data.DataLoader(data,
                    batch_size=self.batch_size,
                    sampler=train_sampler,
                    num_workers=1)

            self.valid_data=torch.utils.data.DataLoader(data,
                    batch_size=self.batch_size,
                    sampler=valid_sampler,
                    num_workers=1)

        elif args[0]=="test":
            data=datasets.CIFAR10(root="./data",train=True,download=True,transform=self.train_transforms)
            self.test_data=torch.utils.data.DataLoader(data, batch_size=self.batch_size,shuffle=False,num_workers=1)



