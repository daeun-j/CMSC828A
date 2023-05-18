from copy import deepcopy, copy
from typing import Dict, OrderedDict

import torch

from fedavg import FedAvgClient
from src.config.utils import *
import gc
from torch.optim import SGD


torch.autograd.set_detect_anomaly(True)

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
        for _ in range(epochs):
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
                
    def fit(self):
        if self.beta == 0 and  self.gb == 0:
            self.beta, self.gb = torch.tensor([self.args.invlmb[self.client_id]]), torch.tensor([0])
        sub_gi, global_model = deepcopy(self.model), deepcopy(self.model)
        for sub_gi_param, pers_param in zip(
                    sub_gi.parameters(),
                    self.pers_model.parameters(),
                    ):
            sub_gi_param.data = sub_gi_param.data - pers_param.data
        cos, self.norm1, self.norm2 = multiply_model(trainable_params(sub_gi), trainable_params(self.gFval_pers_model), normalize=True)
        if self.args.ab == 'b' or self.args.ab == 'B': 
            self.gb = self.args.local_lr * cos
            new_beta = self.beta - self.gb 
            gb = deepcopy(self.gb)
            self.gb = gb
            if new_beta >= 0:
                self.beta = new_beta
                # print(type(self.beta), self.beta, type(self.gb), self.gb)
            self.beta, self.gb = torch.tensor([self.beta]), torch.tensor([self.gb])

        self._fit( global_model, self.trainloader, self.local_epoch)
        self._fit( self.pers_model, self.trainloader, self.local_epoch)

        for gtrg_param, model_param, global_param in zip(self.gFtrg_pers_model.parameters(), 
                                                        self.model.parameters(),
                                                        global_model.parameters(),):
            gtrg_param.data = model_param.data - global_param.data

        # Calculate gFval_i = after - before  theta_i
        pers_bf, pers_af = deepcopy(self.pers_model), deepcopy(self.pers_model) # before  theta_i
        self._fit(pers_af, self.valloader, 20)

        for gFval_param, af_param, bf_param in zip(self.gFval_pers_model.parameters(), pers_af.parameters(), pers_bf.parameters()):
            gFval_param.data = af_param.data - bf_param.data

        gc.collect()
        torch.cuda.empty_cache()
        
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        return super().evaluate(self.pers_model)

    def _evaluate(self) -> Dict[str, Dict[str, float]]:
        return super().evaluate(self.model)