# Local Variables:
# indent-tabs-mode: t
# End:

from typing import Callable, Optional, Dict, Any
import time
import datetime as dt
import copy

import model as m
import evaluation

import numpy as np
from tensorboardX import SummaryWriter  # type: ignore
import torch
import torch.utils.data as tud
import torch.optim as optim
import torch.nn.functional as F
from torchtyping import TensorType as TT  # type: ignore
from torchtyping import patch_typeguard   # type: ignore
from mixmatch import mixmatch
patch_typeguard()
from typeguard import typechecked  # type: ignore

class Timer:
    def __init__(self):
        self.BUFSIZE = 100
        self.buf = np.zeros(self.BUFSIZE)
        self.idx = 0

    def start(self):
        self.time = time.time()

    def stop(self):
        self.buf[self.idx] = time.time() - self.time
        del self.time
        self.idx = (self.idx+1) % self.BUFSIZE

    def query(self):
        return np.mean(self.buf)

class WeightEMA(object):
    def __init__(self, model, lr):
        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.set_training(False)
        for param in self.ema_model.parameters():
            param.detach_()
        self.alpha = 0.999
        self.params = list(self.model.state_dict().values())
        self.ema_params = list(self.ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        with torch.no_grad():
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.params, self.ema_params):
                if ema_param.dtype==torch.float32:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param * one_minus_alpha)
                    # customized weight decay
                    # at every step, lose LR * 0.02. Means param decay of 0.02.
                    param.mul_(1 - self.wd)

class Trainer():
    @typechecked
    def __init__(self, model: m.WideResNet, criterion: Callable,
                 optimizer: optim.Optimizer,
                 labeled_train_dataloader: Any,
                 not_labeled_train_dataloader: Any,
                 val_dataloader: tud.DataLoader,
                 num_epochs: int,
                 results_dir: str,
                 augment: Callable,
                 val_criterion: Callable,
                 mix_match_alpha: Optional[float] = None,
                 mix_match_T: Optional[float] = None,
                 mix_match_K: Optional[int] = None,
                 scheduler : Optional[Any] = None,
             ):
        if not_labeled_train_dataloader is not None:
            assert mix_match_K is not None
            assert mix_match_T is not None
            assert mix_match_alpha is not None
        self.ssl = not_labeled_train_dataloader is not None
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.labeled_train_dataloader = labeled_train_dataloader
        self.not_labeled_train_dataloader = not_labeled_train_dataloader
        self.val_dataloader = val_dataloader
        self.writer = SummaryWriter(results_dir)
        self.results_dir = results_dir
        self.scheduler = scheduler
        self.augment = augment
        self.val_criterion = val_criterion
        self.mix_match_alpha = mix_match_alpha
        self.mix_match_T = mix_match_T
        self.mix_match_K = mix_match_K
        self.ema_opt = WeightEMA(
            self.model,
            self.optimizer.state_dict()['param_groups'][0]['lr'])


        self.data_timer = Timer()
        self.backward_timer = Timer()
        self.forward_timer = Timer()
        self.mixmatch_timer = Timer()
        self.other_timer = Timer()

    @typechecked
    def epoch(self, i: int):
        TRAIN_ITERATION = 1024
        ds = self.not_labeled_train_dataloader if self.ssl else self.labeled_train_dataloader
        self.losses = torch.zeros(TRAIN_ITERATION, requires_grad=False)
        self.accs = torch.zeros(TRAIN_ITERATION, requires_grad=False)
        self.accs_xs = torch.zeros(TRAIN_ITERATION, requires_grad=False)
        self.accs_us = torch.zeros(TRAIN_ITERATION, requires_grad=False)
                
        print(f"Epoch has {TRAIN_ITERATION} batches.")
        i_l_t_d = iter(self.labeled_train_dataloader)
        i_ds = iter(ds)
        start_time_last_20 = time.time()
        for ii in range(TRAIN_ITERATION):

            self.data_timer.start()
            batch = next(i_ds, None)
            if batch is None:
                i_ds = iter(ds)
                batch = next(i_ds)
            if self.ssl:
                us = batch    
            else:
                xs, ys = batch
                ys = ys.to(self.model.device)

            if i == 0 and ii == 0:
                self.writer.add_graph(self.model, torch.zeros(100, 3, 32, 32, requires_grad=False))
            if self.ssl:
                batch = next(i_l_t_d, None)
                if batch is None:
                    i_l_t_d = iter(self.labeled_train_dataloader)
                    batch = next(i_l_t_d)
                xs, ys = batch
                ps = F.one_hot(ys, 10)
                self.data_timer.stop()

                with torch.no_grad():
                    self.mixmatch_timer.start()
                    X_prime, U_prime = mixmatch(X=xs, U=us, T=self.mix_match_T,
                                                alpha=self.mix_match_alpha, 
                                                p=ps, model=self.model, K=self.mix_match_K,
                                                augment = self.augment
                                                )  # type: ignore

                    x, targets_x = X_prime
                    u, targets_u = U_prime
                    self.mixmatch_timer.stop()
                self.optimizer.zero_grad()
                # breakpoint()
                self.forward_timer.start()
                outputs_x, outputs_u = self.model(x), self.model(u)
                loss, preds, ys = self.criterion(outputs_x=outputs_x,
                                                 targets_x=targets_x,
                                                 outputs_u=outputs_u,
                                                 targets_u=targets_u,
                                                 epoch=i + ii / TRAIN_ITERATION)  # type: ignore
                self.forward_timer.stop()
            else:
                zs = self.model(xs)
                loss = self.criterion(outputs_x=zs,
                                      targets_x=ys,
                                      outputs_u=None, targets_u=None,
                                      epoch=i)  # type: ignore
                preds = torch.argmax(zs, dim=1)

            self.backward_timer.start()
            loss.backward()
            self.backward_timer.stop()

            self.other_timer.start()
            self.optimizer.step()
            self.model.eval()
            self.ema_opt.step()
            self.losses[ii] = loss
            if self.ssl:
                x_correct = float((preds == ys)[:x.shape[0]].sum())
                u_correct = float((preds == ys)[x.shape[0]:].sum())
                self.accs_xs[ii] = float(x_correct / x.shape[0])
                self.accs_us[ii] = float(u_correct / u.shape[0])
            else:
                self.accs[ii] = float((preds == ys).to(float).mean())
                        
            if ii % 20 == 0 and ii != 0:
                curr_time = time.time() - start_time_last_20
                start_time_last_20 = time.time()
                u_shape = 0 if not self.ssl else u.shape[0]
                bs: int = self.labeled_train_dataloader.batch_size  # type: ignore
                samples_proc = (i * len(self.labeled_train_dataloader) +
                                ii) * (bs + u_shape)
                samples_proc_last_20 = 20 * (bs + u_shape)
                print(f"Processed {samples_proc_last_20} samples last 20 epochs")
                print(f"batch [{ii} / {TRAIN_ITERATION}], "
                      f"epoch [{i} / {self.num_epochs}], "
                      "train loss:",
                      f"{float(self.losses[ii-20:ii].mean()):.3f} "
                      "train acc (xs): "
                      f"{float(self.accs_xs[ii-20:ii].mean()):.3f} "
                      "train acc (us): "
                      f"{float(self.accs_us[ii-20:ii].mean()):.3f} "
                      f"acc (unlabeled): {float(self.accs[ii-20:ii].mean()):.3f}"
                      )
                print(f"trained on {samples_proc} samples; "
                      f" speed: {samples_proc_last_20/curr_time:.2f} "
                      "samples / sec")
                self.writer.add_scalar(
                    "train/speed",
                    samples_proc_last_20/curr_time,
                    i * TRAIN_ITERATION + ii)
                self.writer.add_scalar(
                        "train/inter_epoch_loss",
                        float(self.losses[ii-20:ii].mean()),
                        i * TRAIN_ITERATION + ii)
                if self.ssl:
                    self.writer.add_scalar(
                        "train/inter_epoch_acc_xs",
                        float(self.accs_xs[ii-20:ii].mean()),
                        i * TRAIN_ITERATION + ii)
                    self.writer.add_scalar(
                        "train/inter_epoch_acc_us",
                        float(self.accs_us[ii-20:ii].mean()),
                        i * TRAIN_ITERATION + ii)
                    self.writer.add_scalar(
                        "train/inter_epoch_Lu",
                        self.criterion.Lu,
                        i * TRAIN_ITERATION + ii
                        )
                    self.writer.add_scalar(
                        "train/inter_epoch_Lx",
                        self.criterion.Lx,
                        i * TRAIN_ITERATION + ii
                        )
                    self.writer.add_scalar(
                        "train/inter_epoch_w",
                        self.criterion.w,
                        i * TRAIN_ITERATION + ii
                        )
                    print(f"Lu = {self.criterion.Lu:.3f}, Lx = {self.criterion.Lx:.3f}, "
                          f"w = {self.criterion.w:.3f}")
                    self.writer.add_scalar(
                        "timer/data",
                        self.data_timer.query(),
                        i * TRAIN_ITERATION + ii)
                    self.writer.add_scalar(
                        "timer/mixmatch",
                        self.mixmatch_timer.query(),
                        i * TRAIN_ITERATION + ii)
                    self.writer.add_scalar(
                        "timer/forward",
                        self.forward_timer.query(),
                        i * TRAIN_ITERATION + ii)
                    self.writer.add_scalar(
                        "timer/backward",
                        self.backward_timer.query(),
                        i * TRAIN_ITERATION + ii)
                    self.writer.add_scalar(
                        "timer/other",
                        self.other_timer.query(),
                        i * TRAIN_ITERATION + ii)
                else:
                    self.writer.add_scalar(
                        "train/inter_epoch_acc",
                        float(self.accs[ii-20:ii].mean()),
                        i * TRAIN_ITERATION + ii)
            self.other_timer.stop()

    @typechecked
    def train(self, resume_epoch: Optional[int]=None) -> Dict[str, TT["num_epochs"]]:
        start_epoch = 0 if resume_epoch is None else resume_epoch + 1
        epoch_mean_train_losses = torch.zeros(self.num_epochs, requires_grad=False)
        epoch_mean_val_losses = torch.zeros(self.num_epochs, requires_grad=False)
        epoch_mean_train_accs = torch.zeros(self.num_epochs, requires_grad=False)
        epoch_mean_val_accs = torch.zeros(self.num_epochs, requires_grad=False)
        for i in range(start_epoch, self.num_epochs):
            self.epoch(i)
            
            if self.scheduler is not None:
                self.scheduler.step()
            #evaluation.log_gradients(i, self.labeled_train_dataloader, next(iter(self.not_labeled_train_dataloader)), 10, self)
            self.model.save_checkpoint(self.results_dir, i)
            epoch_mean_train_losses[i] = torch.mean(self.losses)
            if self.ssl:
                epoch_mean_train_accs[i] = torch.mean(self.accs_xs)
            else:
                epoch_mean_train_accs[i] = torch.mean(self.accs)
            val_stats = self.evaluate(
                 self.ema_opt.ema_model, self.val_dataloader, self.val_criterion)
            epoch_mean_val_losses[i] = val_stats["loss"]
            epoch_mean_val_accs[i] = val_stats["acc"]
            print(f"Average train loss for epoch {i}: "
                  f"{float(epoch_mean_train_losses[i]):.3f}")
            print(f"Average train acc for epoch {i}: "
                  f"{float(epoch_mean_train_accs[i]*100):.3f}%")
            print(f"Val loss after epoch {i}: "
                  f"{float(epoch_mean_val_losses[i]):.3f}")
            print(f"Val acc after epoch {i}: "
                  f"{float(epoch_mean_val_accs[i]*100):.3f}%")
            self.writer.add_scalar("val/loss", val_stats["loss"], i)
            self.writer.add_scalar("val/acc", val_stats["acc"]*100, i)
            self.writer.add_scalar("train/loss", epoch_mean_train_losses[i], i)
            self.writer.add_scalar("train/acc", epoch_mean_train_accs[i]*100, i)
        return {'train_loss': epoch_mean_train_losses,
                'val_loss': epoch_mean_val_losses}

    @typechecked
    def evaluate(self, model: m.WideResNet, on_what: tud.DataLoader, criterion: Callable) -> Dict[str, float]:
        losses = torch.zeros(len(on_what), requires_grad=False)
        accs = torch.zeros(len(on_what), requires_grad=False)
        with torch.no_grad():
            model.set_training(False)
            print(f"Validation size: {len(on_what)} batches")
            for i, (xs, ys) in enumerate(on_what):
                ys = ys.to(model.device)
                zs = model(xs)
                loss = criterion(outputs_x=zs,
                                 targets_x=ys.to(self.model.device), 
                                 outputs_u=None, targets_u=None,
                                 epoch=i)  # type: ignore
                preds = torch.argmax(zs, dim=1)
                tot_correct = float((preds == ys).sum())
                accs[i] = tot_correct / len(ys)
                losses[i] = loss
            model.set_training(True)
        return {"loss": float(losses.mean()), "acc": float(accs.mean())}
