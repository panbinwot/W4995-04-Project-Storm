'''
Pytorch implementation of Stochastic Recursive Momentum 

Example : SOURCE CODE FOR TORCH.OPTIM.SGD
    links: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
'''


import torch
from torch.optim import Optimizer

class Storm(Optimizer):
    def __init__(self):
        pass
    def __setstate__(self,state):
        super(Storm, self).__setstate__(state)
        loss = None
        return loss
