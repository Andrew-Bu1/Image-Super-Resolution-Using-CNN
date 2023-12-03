import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def run_train(self, **kwargs):
        print("Model in training, with args: {}".format(kwargs))

    def run_eval(self, **kwargs):
        print("Model in evaluation, with args: {}".format(kwargs))
