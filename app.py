from argparse import ArgumentParser
from model import Model
import os

args = ArgumentParser()
args.add_argument("--mode",type=str,choices=["train","test"],default="train")
args.add_argument("--epochs",type=int,default=10)
args.add_argument("--ckpts_dir",type=str,default="./ckpts")
args.add_argument("--log_dir",type=str,default="./tboard")
args.add_argument("--verbose_step",type=int,default=100)
args.add_argument("--batch_size",type=int,default=10)
args.add_argument("--lr",type=float,default=0.01)
args.add_argument("--momentum",type=float,default=0.9)

parsed = args.parse_args()

mode = parsed.mode

configs=vars(parsed)
configs["layers"]=[2,2,2,2]

model=Model(**configs)
model.net_summary()
model.train()

print("Success")
