#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Pytorch implementation of Stochastic Recursive Momentum 

Example : SOURCE CODE FOR TORCH.OPTIM.SGD
    links: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
'''


import torch
from typing import *
from torch.optim.optimizer import Optimizer, required

class Storm(Optimizer):
    def __init__(self, params, k=0.1, w=0.1, c=1):
        if k < 0.0:
            raise ValueError("Invalid k")
        if w < 0.0:
            raise ValueError("Invalid w")
        if c is not required and c < 0.0:
            raise ValueError("Invalid c")

        defaults = dict(k=k, w=w, c=c)
        super(Storm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Storm, self).__setstate__(state)

    def norm(self, tensor):
        #TODO
        return torch.norm(tensor)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            k = group['k']
            w = group['w']
            c = group['c']

            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad.data

                param_state = self.state[p]
                if 'storm_d' not in param_state:
                    buf_d = param_state['storm_d'] = torch.clone(d_p).detach()
                    # buf_g = param_state['storm_g'] = self.norm(d_p) ** 2
                    buf_g = param_state['storm_g'] = self.norm(d_p)
                    buf_lr = k / (w ** (1.0 / 3.0))
                    p.data.add_(-buf_lr, buf_d)
                    param_state['storm_momentum'] = c * (buf_lr ** 2)

                else:
                    buf_d = param_state['storm_d']
                    buf_g = param_state['storm_g']
                    buf_momentum = param_state['storm_momentum']

                    # buf_g.add_(self.norm(d_p) ** 2)
                    buf_g.add_(self.norm(d_p))
                    buf_d.add_(d_p).add_((1 - buf_momentum) * (buf_d - d_p))
                    buf_lr = k / ((w + buf_g) ** (1.0 / 3.0))

                    p.data.add_(-buf_lr, d_p)
                    param_state['storm_momentum'] = c * (buf_lr ** 2)
        return loss
