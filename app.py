from argparse import ArgumentParser
from model import ResNet

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
md.build()
if mode == "train":
    md.train()
    md.test()
    md.export()
elif mode == "test":
    md.test()

print("Success")
