from argparse import ArgumentParser
from network import ResNet
import torchvision.datasets as datasets

args = ArgumentParser()
args.add_argument("--mode",type=str,choices=["train","test"],default="train")
args.add_argument("--blocks",type=int,default=5)
args.add_argument("--epochs",type=int,default=10)
args.add_argument("--ckpts",type=str, default="./ckpts")

parsed = args.parse_args()

configs = {"epochs":parsed.epochs,
        "ckpts":parsed.ckpts,
        "blocks":parsed.blocks}
mode = parsed.mode

md = ResNet(configs=configs)

if mode == "train":
    data=datasets.MNIST(root="./data",train=True,download=True,transform=None)
elif mode == "test":
    data=datasets.MNIST(root="./data",train=False,download=True,transform=None)

print("Success")
