from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_peacoc_argparser
from src.config.models import MODEL_DICT
from src.client.peacoc import PeacocClient
import torch
from src.config.utils import *
import torch.nn as nn
import ast
import wandb 
from argparse import ArgumentParser
from src.config.args import get_fedavg_argparser
from torch.nn.functional import normalize
import gc

class PeacocServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "Peacoc",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_peacoc_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        wandb.init(project = self.args.pj, name = self.args.dn)
        
        # init model(s) parameters
        self.model = MODEL_DICT[self.args.model](self.args.dataset).to(self.device)
        self.pers_model = MODEL_DICT[self.args.model](self.args.dataset).to(self.device)
        self.gFval_pers_model = MODEL_DICT[self.args.model](self.args.dataset).to(self.device)
        self.gFtrg_pers_model = MODEL_DICT[self.args.model](self.args.dataset).to(self.device)
        self.pers_model.check_avaliability()
        self.model.check_avaliability()
        self.trainer = PeacocClient(
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total, 
            deepcopy(self.pers_model), deepcopy(self.gFval_pers_model), deepcopy(self.gFtrg_pers_model)
        )
        self.trainable_params_name, init_trainable_params = trainable_params(
            self.model, requires_name=True
        )
        self.global_params_dict: OrderedDict[str, torch.nn.Parameter] = OrderedDict(
            zip(self.trainable_params_name, deepcopy(init_trainable_params))
        )
        self.client_trainable_params: List[List[torch.Tensor]] = [
                deepcopy(init_trainable_params) for _ in self.train_clients
            ]
        self.gFval_trainable_params: List[List[torch.Tensor]] = [
                deepcopy(init_trainable_params) for _ in self.train_clients
            ]
        self.gFtr_trainable_params: List[List[torch.Tensor]] = [
                deepcopy(init_trainable_params) for _ in self.train_clients
            ]
        self.output_betagFval = deepcopy(self.model)
        self.T = self.args.temperature #wandb.config.T #
        self.prealpha = torch.FloatTensor(self.args.lmb)
        self.soft = torch.nn.Softmax(dim=0)
        self.alpha = self.soft(self.prealpha/self.T)  # torch.FloatTensor([1 / self.client_num_in_total] * self.client_num_in_total)
        # self.alpha = self.softmaxwT(self.prealpha, self.T)  # torch.FloatTensor([1 / self.client_num_in_total] * self.client_num_in_total)
        self.ga = torch.FloatTensor([0] * self.client_num_in_total)
        self.global_lr = self.args.global_lr # wandb.config.glr
        self.alpha_list, self.ga_list, self.prealpha_list, self.beta_list, self.gb_list, self.gFtr0_list, self.agg_beta, self.norm1_list, self.norm2_list  = [], [], [], [], [], [], [], [], []
        self.num_list = list(range(self.client_num_in_total))
        self.beta_dict = {i: {} for i in range(self.args.global_epoch)}
        self.gb_dict = {i: {} for i in range(self.args.global_epoch)}
        
    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E
            for param in trainable_params(self.output_betagFval):
                param = torch.zeros_like(param)
            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 20, f"TRAINING EPOCH: {E + 1}", "-" * 20)
            if (E + 1) % self.args.test_gap == 0:
                self.test()
                
            self.selected_clients = self.client_sample_stream[E]
            
            gFval_params_cache = []
            gFtrg_params_cache = []
            beta_weight_cache = []
            client_ids = []
            for client_id in self.selected_clients:
                new_parameters, pers_param, gFval_param, gFtr_param = self.generate_client_params(client_id)
                gFval_m, gFtrg_m, self.beta_dict[E][client_id], self.gb_dict[E][client_id], self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters = new_parameters,
                    pers_param = pers_param,
                    gFval_param = gFval_param, 
                    gFtr_param = gFtr_param,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                gFval_params_cache.append(gFval_m)
                gFtrg_params_cache.append(gFtrg_m)
                beta_weight_cache.append(self.beta_dict[E][client_id])
                client_ids.append(client_id)
            self.aggregate(gFval_params_cache, gFtrg_params_cache, beta_weight_cache, client_ids)
            self.log_info()
            
            
    def test(self):
        loss_before, loss_after = [], []
        correct_before, correct_after = [], []
        num_samples = []

        for client_id in self.test_clients:
            new_parameters, _, _, _ = self.generate_client_params(client_id)
            stats = self.trainer.test(client_id, new_parameters)

            correct_before.append(stats["before"]["test_correct"])
            correct_after.append(stats["after"]["test_correct"])
            loss_before.append(stats["before"]["test_loss"])
            loss_after.append(stats["after"]["test_loss"])
            num_samples.append(stats["before"]["test_size"])
        
        loss_before = torch.tensor(loss_before)
        loss_after = torch.tensor(loss_after)
        correct_before = torch.tensor(correct_before)
        correct_after = torch.tensor(correct_after)
        num_samples = torch.tensor(num_samples)
        
        self.test_results[self.current_epoch + 1] = {
            "loss": "{:.4f} -> {:.4f}".format(
                loss_before.sum() / num_samples.sum(),
                loss_after.sum() / num_samples.sum(),
            ),
            "accuracy": "{:.2f}% -> {:.2f}%".format(
                correct_before.sum() / num_samples.sum() * 100,
                correct_after.sum() / num_samples.sum() * 100,
            ),
        }
        
        
    @torch.no_grad()
    def aggregate(self, gFval_params_cache, gFtrg_params_cache, beta_weight_cache, client_ids):                            
        # Calculate sum beta * gFval_i  
        beta_weight_cache = torch.tensor(beta_weight_cache, device=self.device)
        aggregated_betagFval = [
            torch.sum(beta_weight_cache * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*gFval_params_cache)
        ]
        norm2_list = torch.FloatTensor([0] * self.client_num_in_total)
        for idx, diff in enumerate(gFval_params_cache):
            params = [item for item in diff]
            params2 = beta_weight_cache[idx] * torch.cat([torch.flatten(torch.Tensor(p)) for p in params])
            norm2 = torch.linalg.norm(params2).reshape(1, -1).item()
            # print(norm2, idx, beta_weight_cache[idx], client_ids[idx], self.trainer.beta_dict, beta_weight_cache)
            norm2_list[client_ids[idx]] = norm2
        self.agg_beta.append(norm2_list.clone().to('cpu').tolist())
        norm1_list = torch.FloatTensor([0] * self.client_num_in_total)
        if self.args.ab == 'a' or self.args.ab == 'B': 
            ## sum beta * gFval_i
            for idx, client_id in enumerate(client_ids):
                cos, norm1, _ = multiply_model(gFtrg_params_cache[idx], aggregated_betagFval, normalize=True)
                self.ga[client_id] = cos
                norm1_list[client_id] = norm1
            self.ga = -self.ga * self.args.local_lr * self.args.global_lr * 1e+4
            self.ga = normalize(self.ga, p=1.0, dim = 0)
            self.prealpha = self.alpha - (1/self.T) * self.ga * self.soft(self.alpha/self.T) * (1-self.soft(self.alpha/self.T))
            self.alpha = self.soft(self.prealpha/self.T) 

        self.gFtr0_list.append(norm1_list.clone().to('cpu').tolist())
            
        cur_alpha = torch.index_select(self.alpha, 0, torch.as_tensor(client_ids)).to(self.device)
        aggregated_delta = [
            torch.sum(cur_alpha * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*gFtrg_params_cache)
        ]
        ## aggregate global model
        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.data -= self.args.global_lr * diff.to(self.device)
            
        self.alpha_list.append(self.alpha.clone().to('cpu').tolist())
        self.ga_list.append(self.ga.clone().to('cpu').tolist())
        self.prealpha_list.append(self.prealpha.clone().to('cpu').tolist())
        betas = {k: v for k, v in self.trainer.beta_dict.items()}
        gbs = {k: v[0] for k, v in self.trainer.gb_dict.items()}
        norm1 = {k: v[1] for k, v in self.trainer.gb_dict.items()}
        norm2 = {k: v[2] for k, v in self.trainer.gb_dict.items()}
        
        betas, gbs = list(betas.values()), list(gbs.values())
        norm1, norm2 = list(norm1.values()), list(norm2.values())
        
        self.beta_list.append(betas)
        self.gb_list.append(gbs)
        self.norm1_list.append(norm1)
        self.norm2_list.append(norm2)

        wandb_params_dict = {'beta':self.beta_list, 'gb':self.gb_list, 
                                'alpha':self.alpha_list, 'ga':self.ga_list, 'pre_alpha':self.prealpha_list, 
                                'theta_0-theta_i':self.norm1_list, 'gFval_i':self.norm2_list, 'gFtr0': self.gFtr0_list, 
                                'beta_gFval':self.agg_beta}
        keys = ['client {}'.format(client_id) for client_id in self.num_list]
        for key, value in wandb_params_dict.items():
            wandb.log({key: 
                wandb.plot.line_series( xs=list(range(self.current_epoch + 1)), 
                                        ys=torch.Tensor(value).T,
                                        keys=keys,
                                        title='{}'.format(key),
                                        xname='Global Rounds')})

    
    def generate_client_params(self, client_id: int): # OrderedDict[str, torch.Tensor]:
        return (self.global_params_dict, OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[client_id])
            ),    OrderedDict(
                zip(self.trainable_params_name, self.gFval_trainable_params[client_id])
            ),  OrderedDict(
                zip(self.trainable_params_name, self.gFtr_trainable_params[client_id])
            ))
        
            
    def softmaxwT(self, logits, T):
        max_input = torch.max(logits)
        stabilized_inputs = logits - max_input
        logits = torch.softmax(stabilized_inputs, dim=0)
        logits = logits.tolist()
        logits = [x/T for x in logits]
        bottom = sum([math.exp(x) for x in logits])
        softmax = [max(round(math.exp(x)/bottom, 6), 0) for x in logits]
        return torch.FloatTensor(softmax)

if __name__ == "__main__":
    server = PeacocServer()
    server.run()
    gc.collect()
    torch.cuda.empty_cache()