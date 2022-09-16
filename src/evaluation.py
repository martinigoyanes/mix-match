import collections
import torch.utils.data as tud
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
from torchtyping import TensorType as TT  # type: ignore
from torchtyping import patch_typeguard   # type: ignore
patch_typeguard()
from typeguard import typechecked  # type: ignore
from tensorboardX import SummaryWriter  # type: ignore
import numpy as np

import model as m
import train as tr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@typechecked
def log_gradients(epoch: int, dl: tud.DataLoader, udl: tud.DataLoader, num_batches: int,
          train: "tr.Trainer" ) -> None:
    """
    Logs gradients as tensorboard histograms.
    """
    assert num_batches < len(dl)
    model = train.model
    criterion = train.criterion

    l2s = collections.defaultdict(list)
    
    for i, (xs, ys), (uxs) in zip(range(num_batches), dl, udl):  # type: ignore
        train.optimizer.zero_grad()
        xs = xs.to(DEVICE)
        ys = ys.to(DEVICE)
        uxs = uxs.to(DEVICE)
        loss = criterion(outputs_x=model(xs),
                                        targets_x=ys,
                                        outputs_u=model(uxs),
                                        targets_u=None,
                                        epoch=i)[0]
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                l2s[name].append(
                    float(torch.linalg.norm(param.grad).cpu()))
    for key in l2s:
        train.writer.add_histogram(
            f'train/{key}_l2_grad',
            np.array(l2s[key]),
            epoch)


class MixMatchLoss():
    def __init__(self, epochs: int, lambda_u: float):
        self.epochs = epochs
        self.lambda_u = lambda_u

    def linear_rampup(self, current):
        if self.epochs == 0:
            return 1.0
        else:
            #current = torch.clip((current / self.epochs), min = 0.0, max = 1.0)
            return float(current / self.epochs)

    @typechecked
    def __call__(self, outputs_x: TT[-1, 10], targets_x: TT[-1, 10],
                 outputs_u: TT[-1, 10],
                 targets_u: TT[-1, 10],
                 epoch: float,
                 ) -> Tuple[TT[()], TT[-1], TT[-1]]:
        """targets_x: augmented x targets; not really one-hot, because they
        have been though Mix-Match, but still concentrated.

        targets_u: same thing

        outputs_u: model predictions for U; LOGITS
        outputs_x: model predictions for X; LOGITS

        RETURNS: loss, predicted labels by the model, 'true' labels
        (including the dreamt-up U ones which are not true labels).

        """
        # breakpoint()
        # earch ROW (corresponding to a single sample sample) now sums
        # to 1.
        probs_u : TT[-1, 10] = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u)**2)


        probs_log_x: TT[-1, 10]
        probs_log_x = F.log_softmax(outputs_x, dim=1)

        # Negative average cross-entropy (KL) between exp**probs_log_x and targets_x:
        # let q = exp**probs_log_x, p = targets_x
        # average KL(targets_x, exp**probs_log_x) = average -SUM_label [p(label) * log (q(label) / p(label))]
        # Have to compute this without the assumption that exactly one q(label) is nonzero.

        # This is from MixMatch-pytorch: don't have to fully compute
        # KL(p, p_model), because `KL = p * log(p / p_model) = -p *
        # log(p_model) + CONSTANTS that do not depend on the model parameters.

        Lx = -torch.mean(torch.sum(probs_log_x * targets_x, dim=1))
        # Lx = F.kl_div(probs_log_x, targets_x, reduction='batchmean', log_target=False)
        
        lambda_u = self.lambda_u  * self.linear_rampup(epoch)
        loss = Lx + lambda_u * Lu
        
        preds = torch.cat((torch.argmax(probs_log_x, dim=1),
                           torch.argmax(probs_u, dim=1)),
                          0)
        ys = torch.cat((targets_x, targets_u), 0)
        ys = ys

        self.Lu = Lu
        self.Lx = Lx
        self.w = lambda_u
        
        return loss, preds.detach(), torch.argmax(ys, dim=1).detach()

class Tester():
    def __init__(self):
        pass
    def test(self):
        pass
