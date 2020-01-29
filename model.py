import torch
from network import res_block
import os

class ResNet:
    def __init__(self, configs):
        print("Initialize the model object")
        self.epochs=configs["epochs"]
        self.ckpts=configs["ckpts"]
        self.blocks=configs["blocks"]
        self.batch_size=configs["batch_size"]
        self.input_size=configs["input_size"]
        pass

    def build(self):
        print("Build the model")
        if os.path.exists(self.ckpts):
            print("Trained model exists")
        pass

    def train(self):
        print("Train the model")
        pass

    def test(self):
        print("Test the model")
        with torch.no_grad():
            print("Parameters are not trained")

        pass

    def export(self):
        print("Export the model")
        if not os.path.exists(self.ckpts):
            os.mkdir(self.ckpts)
        pass
