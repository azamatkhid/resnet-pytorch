from argparse import ArgumentParser
from network import ResNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchsummary as summary
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os

args = ArgumentParser()
args.add_argument("--mode",type=str,choices=["train","test"],default="train")
args.add_argument("--epochs",type=int,default=10)
args.add_argument("--ckpts",type=str,default="./ckpts")
args.add_argument("--tboard",type=str,default="./tboard")
args.add_argument("--verbose",type=int,default=0)
args.add_argument("--batch_size",type=int,default=10)
parsed = args.parse_args()

mode = parsed.mode

transform=transforms.Compose([transforms.Resize((224,224),interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net=ResNet(layers=[2,2,2,2])

print(f"Run on {device}")
if torch.cuda.device_count() > 0:
    net=nn.DataParallel(net)
    print(f"Number of GPUs: {torch.cuda.device.count()}")
net.to(device)

if parsed.verbose!=0:
    summary(net,(3,224,224))


if mode == "train":
    data=datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
    dataloader=torch.utils.data.DataLoader(data, batch_size=parsed.batch_size,shuffle=True,num_workers=1)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
     
    if not os.path.exists(parsed.tboard):
        os.mkdir(parsed.tboard)
    writer=SummaryWriter(log_dir=parsed.tboard)
    iteration=0
    for epoch in range(10):
        running_loss=0.0
        for idx, batch in enumerate(dataloader, 0):
            inputs,labels=batch[0].to(device),batch[1].to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            writer.add_scalar("Loss/train",running_loss/parsed.batch_size,iteration)
            iteration+=1
            if idx%parsed.batch_size==(parsed.batch_size-1):
                print("[{:d} {:5d}] loss: {:.3f}".format(epoch+1, idx+1, running_loss/parsed.batch_size))
                running_loss=0.0

    if not os.path.exists(mode.ckpts):
        os.mkdir(mode.ckpts)

    torch.save(net.state_dict,mode.ckpts)


elif mode == "test":
    data=datasets.CIFAR10(root="./data",train=False,download=True,transform=transoform)
    dataloader=torch.utils.data.DataLoader(data, batch_size=10,shuffle=True,num_workers=1)

print("Success")
