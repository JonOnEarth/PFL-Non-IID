# Gaussian mixture federated learning
## bayesian training locally(client)
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time
import copy

class clientbnn(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.inference = args.inference


    def train(self):
        trainloader=self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        # self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # bnn
        elbos = []
        pbar = tqdm(total=max_local_steps, unit="Epochs", postfix=f"Task {i}")

        def callback(_i, _ii, e):
            elbos.append(e / len(trainloader.sampler))
            pbar.update()

        self.model.obs.dataset_size = len(trainloader.sampler)
        optim = pyro.optim.Adam({"lr": 1e-3})
        with tyxe.poutine.local_reparameterization():
            self.model.bnn.fit(trainloader, optim, max_local_steps, device=self.device, callback=callback)

        pbar.close()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        