from fedavg import FedAvgClient
from src.config.utils import trainable_params, calculate_accuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from mango import Tuner
import torch
from mango.domain.distribution import loguniform

class FedProxClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super(FedProxClient, self).__init__(model, args, logger)
        self.df = []
        
    def train(self, client_id, new_parameters, verbose=False):
        delta, _, stats = super().train(
            client_id, new_parameters, return_diff=True, verbose=verbose
        )
        self.client_id = client_id
        return delta, 1.0, stats
    
    # def fit(self):
    #     self.model.train()
    #     global_params = [p.clone().detach() for p in trainable_params(self.model)]
    #     for _ in range(self.local_epoch):
    #         for x, y in self.trainloader:
    #             if len(x) <= 1:
    #                 continue

    #             x, y = x.to(self.device), y.to(self.device)
    #             logit = self.model(x)
    #             loss = self.criterion(logit, y)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             for w, w_t in zip(trainable_params(self.model), global_params):
    #                 # w.grad.data += self.args.mu * (w.data - w_t.data)
    #                 w.grad.data += self.args.invlmb[self.client_id] * (w.data - w_t.data)
    #             self.optimizer.step()

    def fit(self):
        if self.args.fn=='fl' or self.args.fn=='bs':
            self.model.train()
            global_params = [p.clone().detach() for p in trainable_params(self.model)]
            local_loss = []
            local_acc = []
            for i in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    accuracy = calculate_accuracy(logit, y)
                    local_loss.append(loss.item())
                    local_acc.append(accuracy)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for w, w_t in zip(trainable_params(self.model), global_params):
                        w.grad.data += self.args.mu * (w.data - w_t.data)
                    self.optimizer.step() 
            self.model.eval()
            valloss_list = []
            valacc_list = []
            for x, y in self.valloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                valloss = self.criterion(logit, y)
                valaccuracy = calculate_accuracy(logit, y)
                valloss_list.append(valloss.item())
                valacc_list.append(valaccuracy)
            self.df.append([self.client_id, sum(valacc_list) / len(valacc_list), sum(valloss_list) / len(valloss_list), sum(local_acc) / len(local_acc), sum(local_loss) / len(local_loss)])
            df = pd.DataFrame(self.df)
            df.to_csv(f'../{self.args.fn}.csv', index=False)
         
        else:
            conf_dict = dict(num_iteration=5, domain_size=10000, initial_random=3)
            # param_space = {'mu':loguniform(-8, 1)}
            
            # conf_dict = dict(num_iteration=5, domain_size=1000, initial_random=50)
            param_space = {'mu':loguniform(-2, 1)}
            
            # third step: run the optimisation through Tuner
            start_time = time.time()
            tuner = Tuner(param_space, self.objective, conf_dict)  # Initialize Tuner
            optimisation_results = tuner.minimize()
            print(f'The optimisation in series takes {(time.time()-start_time)/60.} minutes.')

            # Inspect the results
            print('best parameters:', optimisation_results['best_params'])
            print('best val loss:', optimisation_results['best_objective'])

            # run the model with the best hyper-parameters on the test set
            df = pd.DataFrame(self.df)
            df.to_csv(f'../{self.args.fn}.csv', index=False)

    # second step: define your objective function
    def objective(self, mu):
        results = []
        global_params = [p.clone().detach() for p in trainable_params(self.model)]
        for hyper_params in mu:    
            local_loss = []
            local_acc = []
            for _ in range(self.local_epoch):
                self.model.train()
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    accuracy = calculate_accuracy(logit, y)
                    local_loss.append(loss.item())
                    local_acc.append(accuracy)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for w, w_t in zip(trainable_params(self.model), global_params):
                        w.grad.data += hyper_params['mu'] * (w - w_t.data)
                    self.optimizer.step()
            self.model.eval()
            valloss_list = []
            valacc_list = []
            for x, y in self.valloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                valloss = self.criterion(logit, y)
                valaccuracy = calculate_accuracy(logit, y)
                valloss_list.append(valloss.item())
                valacc_list.append(valaccuracy)
            if self.args.fn=='by_acc':
                results.append(1/(sum(valacc_list) / len(valacc_list)))
            elif self.args.fn=='by_los':
                results.append(sum(valloss_list) / len(valloss_list))
            self.df.append([self.client_id, sum(valacc_list) / len(valacc_list), sum(valloss_list) / len(valloss_list), sum(local_acc) / len(local_acc), sum(local_loss) / len(local_loss), hyper_params['mu']])
        return results

