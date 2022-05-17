# bayesian_net.py

import torch
import torch.nn as nn
import torch.utils.data as data
import pyro
import pyro.distributions as dist

import numpy as np
import copy
import pandas as pd

from tqdm import tqdm


def conv_3x3(c_in, c_out):
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)


class ConvNet(nn.Sequential):

    def __init__(self):
        super().__init__()
        self.add_module("Conv1_1", conv_3x3(3, 32))
        self.add_module("ReLU1_1", nn.ReLU(inplace=True))
        self.add_module("Conv1_2", conv_3x3(32, 32))
        self.add_module("ReLU1_2", nn.ReLU(inplace=True))
        self.add_module("MaxPool1", nn.MaxPool2d(2, stride=2))

        self.add_module("Conv2_1", conv_3x3(32, 64))
        self.add_module("ReLU2_1", nn.ReLU(inplace=True))
        self.add_module("Conv2_2", conv_3x3(64, 64))
        self.add_module("ReLU2_2", nn.ReLU(inplace=True))
        self.add_module("MaxPool2", nn.MaxPool2d(2, stride=2))

        self.add_module("Flatten", nn.Flatten())

        self.add_module("Linear", nn.Linear(64 * 8 * 8, 512))
        self.add_module("ReLU", nn.ReLU(inplace=True))

        self.add_module("Head", nn.Linear(512, 10))


class FCNet(nn.Sequential):

    def __init__(self):
        super().__init__()
        self.add_module("Linear", nn.Linear(784, 200))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Head", nn.Linear(200, 1))


class bayesnet():
    def __init__(self, net, DEVICE,inference='mean-field'):
        # super().__init__()
        # net.to(DEVICE)
        if inference == "mean-field":
            self.prior = tyxe.priors.IIDPrior(dist.Normal(torch.tensor(0., device=DEVICE), torch.tensor(1., device=DEVICE)),
                                        expose_all=False, hide_modules=[net.Head])
            self.guide = partial(tyxe.guides.AutoNormal, init_scale=1e-4,
                                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net))
            
        elif inference == "ml":
            self.prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True)
            self.guide = None
        else:
            raise RuntimeError("Unreachable")
        
        self.obs = tyxe.likelihoods.Categorical(-1)

        self.net = net
        self.bnn = tyxe.VariationalBNN(self.net,self.prior,self.obs,self.guide)
    


# bnn(net, prior, obs, guide):