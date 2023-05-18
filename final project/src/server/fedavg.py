import pickle
import sys
import json
import os
import random
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

import optuna
from optuna.trial import TrialState
import pdb

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()

sys.path.append(_PROJECT_DIR)

from src.config.utils import OUT_DIR, fix_random_seed, trainable_params
from src.config.models import MODEL_DICT
from src.config.args import get_fedavg_argparser
from src.client.fedavg import FedAvgClient
import numpy as np
import pandas as pd
import wandb
import math
from torch.nn.functional import normalize


        
class FedAvgServer:
    def __init__(
        self,
        algo: str = "FedAvg",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        # retrieving parameters for fedavg 
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = algo
        self.unique_model = unique_model
        fix_random_seed(self.args.seed)
        with open(_PROJECT_DIR / "data" / self.args.dataset / "args.json", "r") as f:
            self.args.dataset_args = json.load(f)
        self.fn = self.args.fn
        # get client party info
        self.train_clients: List[int] = None
        self.test_clients: List[int] = None
        self.client_num_in_total: int = None
        try:
            partition_path = _PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        self.train_clients = partition["separation"]["train"]
        self.test_clients = partition["separation"]["test"]
        self.client_num_in_total = partition["separation"]["total"]

        # init model(s) parameters
        self.device = torch.device(
            "cuda" if self.args.server_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model = MODEL_DICT[self.args.model](self.args.dataset).to(self.device)
        self.model.check_avaliability()
        self.trainable_params_name, init_trainable_params = trainable_params(
            self.model, requires_name=True
        )

        # client_trainable_params is for pFL, which outputs exclusive model per client
        # global_params_dict is for regular FL, which outputs a single global model
        if self.unique_model:
            self.client_trainable_params: List[List[torch.Tensor]] = [
                deepcopy(init_trainable_params) for _ in self.train_clients
            ]
        self.global_params_dict: OrderedDict[str, torch.nn.Parameter] = OrderedDict(
            zip(self.trainable_params_name, deepcopy(init_trainable_params))
        )

        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
       
        if self.args.join_ratio == 1.0:
            self.client_sample_stream = [
                random.sample(
                    self.train_clients, int(self.client_num_in_total * self.args.join_ratio)
                )
                for _ in range(self.args.global_epoch)
            ]
        else:
            self.client_sample_stream = [
            random.sample(
                self.train_clients, int(self.client_num_in_total * self.args.join_ratio)
            )
            for _ in range(self.args.global_epoch)
            ]
 
        self.selected_clients: List[int] = []
        self.current_epoch = 0

        # variables for logging
        if self.args.visible:
            from visdom import Visdom

            self.viz = Visdom()
            self.viz_win_name = (
                f"{self.algo}"
                + f"_{self.args.dataset}"
                + f"_{self.args.global_epoch}"
                + f"_{self.args.local_epoch}"
            )
        self.client_stats = {i: {} for i in self.train_clients}
        self.metrics = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
            "val_before": [],
            "val_after": [],
        }
        self.loss_metrics = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
            "val_before": [],
            "val_after": [],
        }
        self.logger = Console(record=self.args.save_log, log_path=False, log_time=False)
        self.test_results: Dict[int, Dict[str, str]] = {}
        self.train_progress_bar = (
            track(
                range(self.args.global_epoch),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.save_log
            else tqdm(range(self.args.global_epoch), "Training...")
        )

        self.logger.log("=" * 20, "ALGORITHM:", self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))

        # init trainer
        self.trainer = None
        if default_trainer:
            self.trainer = FedAvgClient(deepcopy(self.model), self.args, self.logger)

        # ADD personalization performance metrics
        self.table = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
            "val_before": [],
            "val_after": [],
            }
        self.loss_table = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
            "val_before": [],
            "val_after": [],
            }
        
        
        ## init parameters
        self.args.lmb = [None] * self.client_num_in_total
        self.args.invlmb = [None]

        for cid in range(self.client_num_in_total):
            self.args.lmb[cid] = len(partition["data_indices"][cid]["train"])
        self.args.lmb = torch.FloatTensor(self.args.lmb)
        self.args.lmb = normalize(self.args.lmb, p=1.0, dim = 0) #self.args.lmb**2
        self.args.invlmb = 1/(self.args.lmb)#1/(self.args.lmb**2)
        self.args.invlmb = normalize(self.args.invlmb, p=1.0, dim = 0)
        # set init beta from beysian optmizer
        self.args.invlmb = torch.Tensor([1e-4, 1e-5, 5e-6,  5e-7, 1e-8, 1e-4, 1e-5, 5e-6,  5e-7, 1e-8])
    
    def train(self):
        max_test_acc = 0 
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)
            
            # evaluation during training
            if (E + 1) % self.args.test_gap == 0:
                curr_test_acc = self.test()
                if curr_test_acc > max_test_acc:
                    max_test_acc = curr_test_acc

            self.selected_clients = self.client_sample_stream[E]

            delta_cache = []
            weight_cache = []
            
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                delta, weight, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                weight = self.args.lmb.clone().detach()[client_id]

                delta_cache.append(delta)
                weight_cache.append(weight)
                
            self.aggregate(delta_cache, weight_cache)
            self.log_info()
            #returns the max accuracy obtained from the evaluation during training
            return max_test_acc 
        

      
    def test(self):
        loss_before, loss_after = [], []
        correct_before, correct_after = [], []
        num_samples = []
        for client_id in self.test_clients:
            client_local_params = self.generate_client_params(client_id)
            stats = self.trainer.test(client_id, client_local_params)

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
        return correct_after.sum() / num_samples.sum() * 100

    @torch.no_grad()
    def update_client_params(self, client_params_cache: List[List[torch.nn.Parameter]]):
        if self.unique_model:
            for i, client_id in enumerate(self.selected_clients):
                self.client_trainable_params[client_id] = [
                    param.detach().to(self.device) for param in client_params_cache[i]
                ]
        else:
            raise RuntimeError(
                "FL system don't preserve params for each client (unique_model = False)."
            )

    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        if self.unique_model:
            return OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[client_id])
            )
        else:
            return self.global_params_dict

    @torch.no_grad()
    def aggregate(self, delta_cache: List[List[torch.Tensor]], weight_cache: List[int]):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        # print(delta_cache, type(delta_cache))
        delta_list = [list(delta.values()) for delta in delta_cache]
        aggregated_delta = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*delta_list)
        ]
        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.data -= diff.to(self.device)
            
            
    def check_convergence(self):
        for label, metric in self.metrics.items():
            if len(metric) > 0:
                self.logger.log(f"Convergence ({label}):")
                acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
                min_acc_idx = 10
                max_acc = 0
                for E, acc in enumerate(metric):
                    for i, target in enumerate(acc_range):
                        if acc >= target and acc > max_acc:
                            self.logger.log(
                                "{} achieved {}%({:.2f}%) at epoch: {}".format(
                                    self.algo, target, acc, E
                                )
                            )
                            max_acc = acc
                            min_acc_idx = i
                            break
                    acc_range = acc_range[:min_acc_idx]

    def log_info(self):
        for label in ["train", "test", "val"]:
            # In the `user` split, there is no test data held by train clients, so plotting is unnecessary.
            if (label == "train" and self.args.eval_train) or (label == "test"
                and self.args.eval_test
                and self.args.dataset_args["split"] != "user"
            ) or (label == "val"
                and self.args.eval_val
                and self.args.dataset_args["split"] != "user"
            ):
                correct_before = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"][
                            f"{label}_correct"
                        ]
                        for c in self.selected_clients
                    ]
                )
                correct_after = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["after"][
                            f"{label}_correct"
                        ]
                        for c in self.selected_clients
                    ]
                )
                loss_before = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"][
                            f"{label}_loss"
                        ]
                        for c in self.selected_clients
                    ]
                )
                loss_after = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["after"][
                            f"{label}_loss"
                        ]
                        for c in self.selected_clients
                    ]
                )
                
                num_samples = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"][
                            f"{label}_size"
                        ]
                        for c in self.selected_clients
                    ]
                )

                acc_before = (
                    correct_before.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
                ).item()
                acc_after = (
                    correct_after.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
                ).item()
                
                losses_before = (
                    loss_before.sum(dim=-1, keepdim=True) / num_samples.sum()
                ).item()
                losses_after = (
                    loss_after.sum(dim=-1, keepdim=True) / num_samples.sum()
                ).item()
                
                self.metrics[f"{label}_before"].append(acc_before)
                self.metrics[f"{label}_after"].append(acc_after)
                self.loss_metrics[f"{label}_before"].append(losses_before)
                self.loss_metrics[f"{label}_after"].append(losses_after)
                
                # ADD
                bftable = [self.client_stats[c][self.current_epoch]["before"][f"{label}_correct"]/
                        self.client_stats[c][self.current_epoch]["before"][f"{label}_size"] *100
                            if c in self.selected_clients else 0 for c, _ in enumerate([0]*self.client_num_in_total)]
                bfloss_table = [self.client_stats[c][self.current_epoch]["before"][f"{label}_loss"]/
                        self.client_stats[c][self.current_epoch]["before"][f"{label}_size"]
                            if c in self.selected_clients else 0 for c, _ in enumerate([0]*self.client_num_in_total)]
                table = [self.client_stats[c][self.current_epoch]["after"][f"{label}_correct"]/
                        self.client_stats[c][self.current_epoch]["before"][f"{label}_size"] *100
                            if c in self.selected_clients else 0 for c, _ in enumerate([0]*self.client_num_in_total)]
                loss_table = [self.client_stats[c][self.current_epoch]["after"][f"{label}_loss"]/
                        self.client_stats[c][self.current_epoch]["before"][f"{label}_size"]
                            if c in self.selected_clients else 0 for c, _ in enumerate([0]*self.client_num_in_total)]
                self.table[f"{label}_after"].append(table)
                self.loss_table[f"{label}_after"].append(loss_table)
                self.table[f"{label}_before"].append(bftable)
                self.loss_table[f"{label}_before"].append(bfloss_table)
                
                
                if self.args.visible:
                    self.viz.line(
                        [acc_before],
                        [self.current_epoch],
                        win=self.viz_win_name,
                        update="append",
                        name=f"{label}_acc(before)",
                        opts=dict(
                            title=self.viz_win_name,
                            xlabel="Communication Rounds",
                            ylabel="Accuracy",
                        ),
                    )
                    self.viz.line(
                        [acc_after],
                        [self.current_epoch],
                        win=self.viz_win_name,
                        update="append",
                        name=f"{label}_acc(after)",
                    )
    # run is the objective function for trials
    def run(self, trial = None):  
        if self.trainer is None:
            raise RuntimeError(
                "Specify your unique trainer or set `default_trainer` as True."
            )
        if self.args.visible:
            self.viz.close(win=self.viz_win_name)
        
        args = self.args
        # pdb.set_trace()    
        if trial is not None:
            max_acc = self.train()
            params = {'local_lr': trial.suggest_loguniform('lr', 1e-7, 1e0), 
                      'weight_decay': trial.suggest_float('weight_decay', 0.00, 0.75,step =0.005),
                      'momentum': trial.suggest_float('momentum', 0.9, 0.99,step =0.01)}
            # update_args(args,params)
        else:
            max_acc = self.train()

        self.logger.log(
            "=" * 20, self.algo, "TEST RESULTS:", "=" * 20, self.test_results
        )
        self.check_convergence()

        # save log files
        if not os.path.isdir(OUT_DIR /  self.args.dataset / self.algo) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(OUT_DIR /  self.args.dataset / self.algo, exist_ok=True)

        if self.args.save_log:
            self.logger.save_text(OUT_DIR /  self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_log.html")

        if self.args.save_metrics:
            import pandas as pd
            import numpy as np

            accuracies = []
            labels = []
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    accuracies.append(np.array(acc).T)
                    labels.append(label)
            pd.DataFrame(np.stack(accuracies, axis=1), columns=labels).to_csv(
                OUT_DIR /  self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_acc_metrics.csv",
                index=False,
            )
            
            losses = []
            labels = []
            for label, loss in self.loss_metrics.items():
                if len(loss) > 0:
                    losses.append(np.array(loss).T)
                    labels.append(label)
            pd.DataFrame(np.stack(losses, axis=1), columns=labels).to_csv(
                OUT_DIR /  self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_loss_metrics.csv",
                index=False,
            )
            
            # ADD
            for label, acc in self.table.items():
                if 'after' in label:
                    pd.DataFrame(np.array(self.table[label][1:])).to_csv(
                            OUT_DIR /  self.args.dataset/ self.algo / f"client_gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_{label}_acc_metrics.csv",
                            index=False,
                        )
            for label, acc in self.loss_table.items():
                if 'after' in label:
                    pd.DataFrame(np.array(self.loss_table[label][1:])).to_csv(
                            OUT_DIR /  self.args.dataset / self.algo / f"client_gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_{label}_loss_metrics.csv",
                            index=False,
                        )
                
        if self.args.save_fig and (self.args.global_epoch != 0):
            import matplotlib
            from matplotlib import pyplot as plt

            matplotlib.use("Agg")
            linestyle = {
                "test_before": "solid",
                "test_after": "solid",
                "train_before": "dotted",
                "train_after": "dotted",
                "val_before": "dashed",
                "val_after": "dashed",
            }
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    plt.plot(acc, label=label, ls=linestyle[label])
            plt.title(f"{self.algo} {self.args.dataset} acc")
            plt.ylim(0, 100)
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.grid()
            plt.legend()
            plt.savefig(
                OUT_DIR / self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_acc.jpeg", bbox_inches="tight"
            )
            plt.clf()
            _min, _max = 1e+4, 0
            for label, loss in self.loss_metrics.items():
                if len(loss) > 0:
                    cur_min_loss, cur_max_loss = np.min(loss), np.max(loss)
                    plt.plot(loss, label=label, ls=linestyle[label])
                    _min, _max = min(_min, cur_min_loss), max(_max, cur_max_loss)
                    
            plt.title(f"{self.algo} {self.args.dataset} loss")

            plt.axis('auto')
            
            plt.xlabel("Communication Rounds")
            plt.ylabel("Loss")
            plt.grid()
            plt.legend()
            plt.savefig(
                OUT_DIR /  self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_loss.jpeg", bbox_inches="tight"
            )
            plt.clf()
            
            gr = self.args.global_epoch
            le = self.args.local_epoch
        
        # raise exception for pruned or NaN trials
        bad_trial = np.isnan(max_acc)
        if trial.should_prune() or bad_trial:
            raise optuna.exceptions.TrialPruned()
        trial.report(max_acc, self.args.global_epoch)
        
        # run function is the objective function to be maximized
        return max_acc 
        #     for label, acc in self.table.items():
        #         if len(acc) > 0:
        #             acc = np.array(self.table[label][1:])
        #             acc = np.where(acc == 0, np.nan, acc)
        #             acc = pd.DataFrame(acc)
        #             acc = acc.T.agg(['mean', 'std'])
        #             plt.plot(np.linspace(1, gr*le, num=int(gr)-1), acc.iloc[0], ls=linestyle[label], label=label)
        #             plt.fill_between(np.linspace(1, gr*le, num=int(gr)-1), acc.iloc[0] - acc.iloc[1],  acc.iloc[0] + acc.iloc[1], alpha=0.2)
        #     plt.legend()
        #     plt.title(f"{self.algo} {self.args.dataset} {'personal acc'}")
        #     plt.ylim(0, 100)
        #     plt.xlabel("# of global rounds * # of local epochs")
        #     plt.ylabel(f"GR:{gr}, LE{le} Accuracy")
        #     plt.grid()
        #     plt.legend()
        #     plt.savefig(
        #         OUT_DIR /  self.args.dataset / self.algo / f"Clients_gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_acc.jpeg", bbox_inches="tight"
        #     )
        #     plt.clf()
        #     _min, _max = 1e+4, 0
        #     for label, loss in self.loss_table.items():
        #         if len(loss) > 0:
        #             loss = np.array(self.loss_table[label][1:])
        #             loss = np.where(loss == 0, np.nan, loss)
        #             cur_min_loss, cur_max_loss = np.min(loss), np.max(loss)
        #             loss = pd.DataFrame(loss)
        #             loss = loss.T.agg(['mean', 'std'])
        #             plt.plot(np.linspace(1, gr*le, num=int(gr)-1), loss.iloc[0], ls=linestyle[label], label=label)
        #             plt.fill_between(np.linspace(1, gr*le, num=int(gr)-1), 
        #                                 loss.iloc[0] - loss.iloc[1], loss.iloc[0] + loss.iloc[1], alpha=0.2)
        #             _min, _max = min(_min, cur_min_loss), max(_max, cur_max_loss)
        #     plt.legend()
        #     plt.title(f"{self.algo} {self.args.dataset} {'personal loss'}")
        #     plt.axis('auto')

        #     plt.xlabel("# of global rounds * # of local epochs")
        #     plt.ylabel(f"GR:{gr}, LE{le} Loss")
        #     plt.grid()
        #     plt.legend()
        #     plt.savefig(
        #         OUT_DIR / self.args.dataset / self.algo / f"Clients_gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.fn}_{self.args.ab}_loss.jpeg", bbox_inches="tight"
        #     )
        #     plt.clf()
        

        # # save trained model(s)
        # if self.args.save_model:
        #     model_name = (
        #         f"{self.args.dataset}_{self.args.global_epoch}_{self.args.model}_{self.args.ab}.pt"
        #     )
        #     if self.unique_model:
        #         torch.save(
        #             self.client_trainable_params, OUT_DIR / self.algo / model_name
        #         )
        #     else:
        #         torch.save(self.model.state_dict(), OUT_DIR / self.algo / model_name)




def update_args(args,params):
    # updates args in place
    dargs = vars(args)
    dargs.update(params) # updates the corresponding (key,value), otherwise adds a new key with correspondong params
    
    
import logging
import sys



if __name__ == "__main__":
    server = FedAvgServer()
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study-cifar100-fedavg"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                load_if_exists = True,
                                sampler = optuna.samplers.TPESampler(), 
                                pruner= optuna.pruners.MedianPruner(),
                                direction='maximize')
    
    study.optimize(server.run, n_trials=100, gc_after_trial = True)
    
    # Summary
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    
    # server.run()
    