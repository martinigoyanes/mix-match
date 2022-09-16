from typing import Tuple, Callable, List
from torchtyping import TensorType as TT  # type: ignore
import torch
import numpy as np
import torch.nn.functional as F
from torchtyping import patch_typeguard   # type: ignore
patch_typeguard()
from typeguard import typechecked  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@typechecked
def sharpen(p: TT[-1, 10], T: float) -> TT[-1, 10]:
    num = (p**(1/T))
    denom = num.sum(dim = 1, keepdim=True)
    return  num / denom

@typechecked
def mixmatch(X: TT[-1, 3, 32, 32],
             U: TT[-1, 3, 32, 32],
             T: float, alpha: float, p: TT[-1, 10],
             model: Callable, K: int,
             augment: Callable
             ) -> Tuple[
                 # TODO read up on types to make this simpler. Want to write
                 # BatchT[D.batch_size], BatchT[D.batch_size*K]
                 Tuple[TT[-1, 3, 32, 32], TT[-1, 10]],
                 Tuple[TT[-1, 3, 32, 32], TT[-1, 10]]
             ]:
    """Our implementation of Algorithm 1 on page 4 of the Mix Match
    article.

    model assumed to take batches TT[-1, 3, 32, 32] and returns logit
    probs TT[-1, 10].  `p` is one-hot labels for the X input.

    Returns AUGMENTED samples and corresponding TARGET PROBABILITIES.
    Written to be used with evaluation.MixMatchLoss.

    """
    model.set_training(False)  # type: ignore
    batch_size = X.shape[0]
    x_hat = torch.zeros(X.shape, requires_grad=False)
    u_hat = torch.zeros((batch_size, K, 3, 32, 32), requires_grad=False)
    qhat = torch.zeros(batch_size, K, 10, requires_grad=False)

    # This part is inspired by
    # https://github.com/YU1ut/MixMatch-pytorch/. Copied it over when
    # searching for why we got worse results (it wasn't our MixMatch),
    # kept it because it was much faster and more readable.
    assert K == 2
    with torch.no_grad():
        inputs_u = augment(U)
        inputs_u2 = augment(U)
        outputs_u = model(inputs_u.to(DEVICE))
        outputs_u2 = model(inputs_u2.to(DEVICE))
        u_p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        targets_u = sharpen(u_p, T).detach()
        all_inputs = torch.cat([X, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([p.to(DEVICE), targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        X_inputs = mixed_input[:batch_size]
        X_targets = mixed_target[:batch_size]

        U_inputs = mixed_input[batch_size:]
        U_targets = mixed_target[batch_size:]
        model.set_training(True)  # type: ignore
    return (X_inputs, X_targets), (U_inputs, U_targets)

if __name__ == "__main__":
    import dataset
    data = dataset.MixMatchData(num_workers=0)
    dataloaders = data.get_dataloaders(labeled_prop=0.005, val_prop=0.2, batch_size=5, augment=True)
    xs, ys = next(x for x in dataloaders["train_labeled"])
    us = next(x for x in dataloaders["train_not_labeled"])
    ps = F.one_hot(ys, 10)

    @typechecked
    def fake_model(t: TT["size": ..., 3, 32, 32]) -> TT["size":..., 10]:
        return torch.ones(t.shape[0], 10) * 0.1

    output = mixmatch(xs, us, 0.5, 0.5, ps, fake_model, K=2)
    print(output)
