from model import WideResNet
from train import Trainer
from evaluation import MixMatchLoss, Tester
import dataset

import pathlib
import datetime as dt
from typing import Dict, Any
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary  # type: ignore

from torchtyping import TensorType as TT  # type: ignore
from torchtyping import patch_typeguard   # type: ignore
patch_typeguard()
from typeguard import typechecked  # type: ignore

time_at_start = dt.datetime.now()

def get_data_dir(params: Dict) -> str:
        time_string = time_at_start.strftime("%d_%H_%M")
        data_dir = (dataset.root / "results" /
                       f"{params['experiment_name']}_{time_string}")
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir)

def get_latest_data_dir(params: Dict) -> str:
        p = dataset.root / "results"
        dirs_p = [x for x in p.iterdir() if
                  x.is_dir() and
                  x.parts[-1].startswith(params["experiment_name"])]

        if not dirs_p:
                assert False, "Can't resume if there are no runs"

        # Get the one that changed last.
        latest_dir = max(dirs_p, key=lambda x: x.stat().st_mtime)

        dirs_p_verbose = ' '.join(x.parts[-1] for x in dirs_p)
        print(f"Path {p} contains runs {dirs_p_verbose}. Latest run is "
              f"{latest_dir.parts[-1]}.")
        return str(latest_dir)

def save_train_params(params: Dict[str, Any]):
        params_path = pathlib.Path(get_data_dir(params)) / 'train_params.txt'
        with open(params_path, 'w') as fl:
                print(params, file=fl)

def run_experiment(params: Dict[str, Any], resume=False) -> None:
        data = dataset.MixMatchData(num_workers=0)
        ce = nn.CrossEntropyLoss()
        @typechecked
        def val_criterion(outputs_x: TT[-1, 10], targets_x: TT[-1],
                      **kwargs) -> TT["batch": ...]:
                return ce(outputs_x, targets_x)

        if params['SSL']:
                dataloaders = data.get_dataloaders(**params['data'])
                criterion = MixMatchLoss(**params["criterion"])

        else:
                dataloaders = data.get_labeled_dataloaders(**params['data'])
                criterion = val_criterion

        if resume:
                results_dir = get_latest_data_dir(params)
        else:
                results_dir = get_data_dir(params)
                save_train_params(params)

        model = WideResNet(**params["network"])
        if resume:
                epoch = model.load_latest_checkpoint(results_dir)

        print(model)
        summary(model, (3, 32, 32))
        optimizer = optim.Adam(model.parameters(), **params["optimizer"])

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     **params["scheduler"])
        if resume:
                optimizer.zero_grad()
                optimizer.step()
                for _ in range(epoch+1):
                        scheduler.step()

        trainer = Trainer(**params["training"], model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          labeled_train_dataloader=dataloaders["train_labeled"],
                          not_labeled_train_dataloader=dataloaders["train_not_labeled"],
                          val_dataloader=dataloaders["val"],
                          results_dir = results_dir,
                          scheduler = scheduler,
                          augment = data.augment,
                          val_criterion = val_criterion
                          )
        # Inside training the MixMatchAlgo gets called and the
        # unlabelled augmentation + guessing occurs
        if resume:
                trainer.train(resume_epoch=epoch)
        else:
                trainer.train()


def fully_supervised_baseline():
        '''Baseline model for the fully supervised/labeled case'''
        params = {
                "experiment_name": "wrn-28-10_4k_data",
                "SSL": False,
                "data": {"val_prop": 0.05, "batch_size": 100, "augment": True},
                "network": {"numClasses": 10, "depth": 28,
                            "widen_factor": 10, "dropoutRate": 0.2},
                "optimizer": {"lr": 0.0001 #, "weight_decay": 0.0005
                              },
                "training": {"num_epochs": 2000},
                "criterion": {"epochs": 200, "lambda_u": 0},
                "scheduler": {"gamma": 0.989, "verbose": True}
        }
        run_experiment(params, resume=False)

def fully_supervised_medium():
        '''Baseline model for the fully supervised/labeled case'''
        params = {
                "experiment_name": "wrn-10-10",
                "SSL": False,
                "data": {"val_prop": 0.2, "batch_size": 100, "augment": True},
                "network": {"numClasses": 10, "depth": 10,
                            "widen_factor": 10, "dropoutRate": 0.0},
                "optimizer": {"lr": 0.001 #, "weight_decay": 0.0005
                              },
                "training": {"num_epochs": 200},
                "criterion": {},
                "scheduler": {"gamma": 0.9, "verbose": True}
        }
        run_experiment(params, resume=True)


def experiment_1():
        '''
        Evaluate how Semi-supervised (MixMatch) performance degrades
        as the proportion of labelled to unlabelled training examples
        changes drastically.

        Total training examples: 50000
        Splits to be evaluated: 
                1) 250 labeled  (0.005), 49750 unlabeled
                2) 500 labeled  (0.010), 49500 unlabeled
                3) 1000 labeled (0.020), 49000 unlabeled
                4) 2000 labeled (0.040), 48000 unlabeled
                5) 4000 labeled (0.080), 46000 unlabeled
        '''
        val_prop = 0.01
        train_size = 50000 * (1-val_prop)
        def labeled_prop(target_labeled: int) -> float:
            return target_labeled / train_size
        params = {
                "experiment_name": "ssl-wrn-28-2-1000",
                "SSL": True,
                "data": {"labeled_prop": labeled_prop(1000), "val_prop": val_prop,
                         "batch_size": 64, "augment": True},
                "network": {"numClasses": 10, "depth": 28,
                            "widen_factor": 2, "dropoutRate": 0.0,
                            "momentum": 0.001},
                "optimizer": {"lr": 0.002 },
                "training": {"num_epochs": 50, "mix_match_alpha": 0.75,
                             "mix_match_T": 0.5, "mix_match_K": 2
                             },
                "criterion": {"epochs": 1024, "lambda_u": 75},
                "scheduler": {"gamma": 1., "verbose": True}
        }
        run_experiment(params, resume=True)
