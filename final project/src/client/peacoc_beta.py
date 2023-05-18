from copy import deepcopy, copy
from typing import Dict, OrderedDict

import torch
torch.cuda.empty_cache()
from fedavg import FedAvgClient
from src.config.utils import *
import gc
from torch.optim import SGD
import pandas as pd
import os.path
torch.autograd.set_detect_anomaly(True)
from torch.optim import optim
import torch.nn as nn
import torch.nn.functional as F

    
class PeacocClient(FedAvgClient):
    def __init__(self, model, args, logger, client_num, pers_m, gFval_m, gFtrg_m):
        super().__init__(model, args, logger)
        self.pers_model = deepcopy(model) # theta_i
        self.gFval_pers_model = deepcopy(model) # init gradient of F val personal model
        self.gFtrg_pers_model = deepcopy(model) # init gradient of F train global model
        self.norm1, self.norm2 = torch.FloatTensor([0]), torch.FloatTensor([0])
        self.personal_params_dict = {
            cid: deepcopy(self.pers_model.state_dict()) for cid in range(client_num)
        }
        self.optimizer.add_param_group(
            {"params": trainable_params(self.pers_model), "lr": self.local_lr}
        ) 

        self.gFval_pers_model_params_dict = {
            cid: deepcopy(self.gFval_pers_model.state_dict()) for cid in range(client_num)
        }
        self.gFtrg_pers_model_params_dict = {
            cid: deepcopy(self.gFtrg_pers_model.state_dict()) for cid in range(client_num)
        }

        self.beta, self.gb = torch.tensor([0]), torch.tensor([0])
        self.nlayers = num_layers(model)
        self.beta_dict = {
            cid: self.beta for cid in range(client_num)
        }
        self.gb_dict = {
            cid: (self.gb, self.norm1.item(), self.norm2.item()) for cid in range(client_num)
        }
                   
    def save_state(self):
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())
        self.beta_dict[self.client_id] = deepcopy(self.beta) 
        self.gb_dict[self.client_id] = (deepcopy(self.gb), self.norm1, self.norm2)
        self.personal_params_dict[self.client_id] = deepcopy(self.pers_model.state_dict())
        self.gFval_pers_model_params_dict[self.client_id] = deepcopy(self.gFval_pers_model.state_dict())
        self.gFtrg_pers_model_params_dict[self.client_id] = deepcopy(self.gFtrg_pers_model.state_dict())

    def set_parameters(self, new_parameters, pers_param=None, gFval_param = None, gFtr_param=None):
        pers_param = self.personal_params_dict[self.client_id]
        gFval_param = self.gFval_pers_model_params_dict[self.client_id]
        gFtr_param = self.gFtrg_pers_model_params_dict[self.client_id]
        self.beta = self.beta_dict[self.client_id]
        self.gb = self.gb_dict[self.client_id][0] 
        self.model.load_state_dict(new_parameters, strict=False)
        self.pers_model.load_state_dict(pers_param, strict=False) # load theta_i
        self.gFtrg_pers_model.load_state_dict(gFtr_param, strict=False) # load theta_i
        self.gFval_pers_model.load_state_dict(gFval_param, strict=False) # load theta_i

    def train(
        self,
        client_id: int,
        new_parameters,
        pers_param,
        gFval_param,
        gFtr_param, 
        verbose=False,):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters=new_parameters, pers_param=pers_param, gFval_param=gFval_param, gFtr_param=gFtr_param, )
        eval_stats = self.train_and_log(verbose=verbose)
        return( deepcopy(trainable_params(self.gFval_pers_model)),
                deepcopy(trainable_params(self.gFtrg_pers_model)), 
                deepcopy(torch.tensor([self.beta])),
                deepcopy(torch.tensor([self.gb])),
                eval_stats,
                )

    def _fit(self, training_model, dataloader, epochs):
        training_model.train()
        optimizer = SGD(
            trainable_params(training_model),
            self.local_lr,
            self.args.momentum,
            self.args.weight_decay,
        ) 
        out_loss = []
        for _ in range(epochs):
            losslist = []
            for x, y in dataloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logit = training_model(x)
                loss = self.criterion(logit, y)
                optimizer.zero_grad()
                loss.backward()
                if training_model == self.pers_model:
                    for pers_param, global_param in zip(
                        trainable_params(training_model),
                        trainable_params(self.model),
                        ):
                        beta = self.beta.item()
                        pers_param_copy = beta * (pers_param.clone() - global_param.clone())
                        pers_param_grad_copy = pers_param.grad.clone()  # make a copy of pers_param
                        pers_param.grad.data = pers_param_grad_copy + pers_param_copy # assign the modified copy back to pers_param `
                optimizer.step()
                losslist.append(loss.item()/len(dataloader))
            out_loss.append(np.mean(losslist))
        return(out_loss)
            
                
    def fit(self):
        self.beta, self.gb = torch.tensor([self.args.temperature]), torch.tensor([0])
        pers_loss = self._fit( self.pers_model, self.trainloader, 20)
        val_loss = self._fit(self.pers_model, self.valloader, 20)
        df = pd.DataFrame({"id":self.client_id,
                            "val_loss": val_loss, 
                            "pers_loss": val_loss, })
        if os.path.isfile(str(self.args.temperature)+'_test_loss.csv'):
            df_bf = pd.read_csv(str(self.args.temperature)+'_test_loss.csv')
            df_bf = pd.concat([df_bf, df], axis=0)
            df_bf.to_csv(str(self.args.temperature)+'_test_loss.csv', index=False)
            
        else:
            df.to_csv(str(self.args.temperature)+'_test_loss.csv', index=False)
        gc.collect()
        torch.cuda.empty_cache()
        
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        return super().evaluate(self.pers_model)

    def _evaluate(self) -> Dict[str, Dict[str, float]]:
        return super().evaluate(self.model)
    