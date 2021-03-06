import os
import torch
import numpy as np

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

import model
from model import ResNet
from tqdm import tqdm

import hydra
from hydra import utils
from omegaconf import DictConfig

class Application:
    def __init__(self,cfg):
        self.cfg=cfg
        self.dataset=getattr(datasets,self.cfg.dataset.upper())
        self.data_dir=os.path.join(utils.get_original_cwd(),"./data")
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.train_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5])])
        
        self.test_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5])])

        self.net=None

    def build(self):
        block=getattr(model,self.cfg.block)
        self.net=ResNet(layers=self.cfg.layers,res_block=block,num_classes=self.cfg.num_classes)
        
        if torch.cuda.device_count()>0:
            self.net=nn.DataParallel(self.net)
            print(f"Number of GPUs {torch.cuda.device_count()}")
        self.net.to(self.device)

        if self.cfg.verbose and torch.cuda.device_count()<=1:
            summary(self.net,(3,224,224))


    def train(self, criterion=nn.CrossEntropyLoss,optimizer=torch.optim.SGD):
        self.net.train() # required due to BN layer
        self._check_dirs()
        self._load_data("train")

        self.writer=SummaryWriter(log_dir=self.cfg.log_dir)
        self.criterion=criterion()
        self.optimizer=optimizer(self.net.parameters(),lr=self.cfg.lr,momentum=self.cfg.momentum,weight_decay=self.cfg.weight_decay)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.cfg.lr_step,gamma=self.cfg.lr_gamma)

        iteration=1
        for epch in range(self.cfg.epochs):
            running_loss=0.0
            epch_loss=0.0
            for idx, batch in enumerate(self.train_data,start=0):
                inputs,labels=batch[0].to(self.device),batch[1].to(self.device)
                self.optimizer.zero_grad()
                outputs=self.net(inputs)
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
                epch_loss+=loss.item()

                if idx%self.cfg.verbose_step==self.cfg.verbose_step-1:
                    valid_acc, valid_loss=self._validation()
                    self.writer.add_scalar("Loss/Train",running_loss/self.cfg.verbose_step,iteration)
                    self.writer.add_scalar("Loss/Validation",valid_loss,iteration)
                    self.writer.add_scalar("Acc/Validation",valid_acc,iteration)
                    self.writer.add_scalar("LearningRate",self.scheduler.get_lr()[0],iteration)
                    print(f"{epch} train_loss: {running_loss/self.cfg.verbose_step}, val_loss: {valid_loss}, val_acc: {valid_acc}, lr: {self.scheduler.get_lr()[0]}")
                    running_loss=0.0
                    iteration+=1
            
            self.scheduler.step()

        torch.save(self.net.state_dict(),os.path.join(self.cfg.ckpts_dir,"model.pth"))

    def _validation(self):
        self.net.eval() # required due to BN layer
        correct=0
        total=0
        total_loss=0
        with torch.no_grad():
            for data in self.valid_data:
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
        self.net.train()


    def _check_dirs(self):
        if not os.path.exists(self.cfg.log_dir):
            os.mkdir(self.cfg.log_dir)
        if not os.path.exists(self.cfg.ckpts_dir):
            os.mkdir(self.cfg.ckpts_dir)

    def _load_data(self,*args):
        if args[0]=="train":
            data=self.dataset(root=self.data_dir,
                    train=True,download=True,transform=self.train_transforms)

            data_size=len(data)
            indices=list(range(data_size))
            valid_ratio=0.1
            split=int(np.floor(valid_ratio*data_size))
            train_idx,valid_idx=indices[split:],indices[:split]
            train_sampler=SubsetRandomSampler(train_idx)
            valid_sampler=SubsetRandomSampler(valid_idx)

            self.train_data=torch.utils.data.DataLoader(data,
                    batch_size=self.cfg.batch_size,
                    sampler=train_sampler,
                    num_workers=1)

            self.valid_data=torch.utils.data.DataLoader(data,
                    batch_size=self.cfg.batch_size,
                    sampler=valid_sampler,
                    num_workers=1)

        elif args[0]=="test":
            data=self.dataset(root=self.data_dir,
                    train=False,download=True,transform=self.test_transforms)
            self.test_data=torch.utils.data.DataLoader(data, batch_size=self.cfg.batch_size,shuffle=False,num_workers=1)

@hydra.main(config_path="./default.yaml")
def main(cfg: DictConfig) -> None:
    app=Application(cfg.parameters)
    app.build()
    app.train()
    app.test()

    print("Success")


if __name__=="__main__":
    main()
