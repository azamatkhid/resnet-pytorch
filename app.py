import os
from argparse import ArgumentParser

from model import Model
from network import BasicBlock, Bottleneck

args = ArgumentParser()
args.add_argument("--mode",type=str,choices=["train","test"],default="train")
args.add_argument("--epochs",type=int,default=20)
args.add_argument("--ckpts_dir",type=str,default="output/ckpts")
args.add_argument("--log_dir",type=str,default="output/tboard")
args.add_argument("--verbose_step",type=int,default=100)
args.add_argument("--verbose",type=int,default=1)
args.add_argument("--batch_size",type=int,default=10)
args.add_argument("--lr",type=float,default=0.001)
args.add_argument("--momentum",type=float,default=0.9)

parsed = args.parse_args()

mode = parsed.mode

configs=vars(parsed)
configs["layers"]=[2,2,2,2]
configs["block"]=BasicBlock

model=Model(**configs)
model.train()
model.test()

print("Success")
