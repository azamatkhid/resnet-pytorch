from argparse import ArgumentParser
from network import ResNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchsummary as summary
import torch.nn as nn
import torch

args = ArgumentParser()
args.add_argument("--mode",type=str,choices=["train","test"],default="train")
args.add_argument("--blocks",type=int,default=5)
args.add_argument("--epochs",type=int,default=10)
args.add_argument("--ckpts",type=str, default="./ckpts")
args.add_argument("--verbose",type=int,default=0)
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
    dataloader=torch.utils.data.DataLoader(data, batch_size=10,shuffle=True,num_workers=1)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)

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
            if idx%10==9:
                print("[{:d} {:5d}] loss: {:.3f}".format(epoch+1, idx+1, running_loss/10))
                running_loss=0.0
    torch.save(net.state_dict,mode.ckpts)


elif mode == "test":
    data=datasets.CIFAR10(root="./data",train=False,download=True,transform=transoform)
    dataloader=torch.utils.data.DataLoader(data, batch_size=10,shuffle=True,num_workers=1)




print("Success")
