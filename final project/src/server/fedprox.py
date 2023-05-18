from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedprox_argparser
from src.client.fedprox import FedProxClient


import optuna
from optuna.trial import TrialState
import pdb


class FedProxServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedProx",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedprox_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedProxClient(deepcopy(self.model), self.args, self.logger)



def update_args(args,params):
    # updates args in place
    dargs = vars(args)
    dargs.update(params) # updates the corresponding (key,value), otherwise adds a new key with correspondong params
    
    
import logging
import sys

if __name__ == "__main__":
    server = FedProxServer()
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study-cifar100-fedprox"  # Unique identifier of the study.
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
