from fedavg import FedAvgClient
from src.config.utils import trainable_params
import numpy as np
import math
import torch 

class FedProxClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super(FedProxClient, self).__init__(model, args, logger)

    def train(self, client_id, new_parameters, verbose=False):
        delta, _, stats = super().train(
            client_id, new_parameters, return_diff=True, verbose=verbose
        )
        self.client_id = client_id
        return delta, self.args.invlmb[self.client_id], stats
        # return delta, 1.0, stats

    # FedProx lambda
    def softmaxwT(self, logits, T):
        max_input = torch.max(logits)
        stabilized_inputs = logits - max_input
        logits = torch.softmax(stabilized_inputs, dim=0)
        logits = logits.tolist()
        logits = [x/T for x in logits]
        bottom = sum([math.exp(x) for x in logits])
        softmax = [max(round(math.exp(x)/bottom, 6), 0) for x in logits]
        return torch.FloatTensor(softmax)
    
    def fit(self):
        self.model.train()
        global_params = [p.clone().detach() for p in trainable_params(self.model)]
        for i in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for w, w_t in zip(trainable_params(self.model), global_params):
                    w.grad.data += self.args.invlmb[self.client_id] * (w.data - w_t.data)
                    # w.grad.data += self.args.mu * (w.data - w_t.data)
                self.optimizer.step()
        # print('fit', self.args.prox_lambda, self.args.lmb[self.client_id] * (1/(i+1)))
