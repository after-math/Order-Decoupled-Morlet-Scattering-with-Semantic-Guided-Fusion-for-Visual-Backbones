"""B-recipe arithmetic, deterministic global Mixup, and fixed200-epoch schedule."""
import math
import numpy as np
import torch
import torch.nn.functional as F

def mixup_plan(epoch, update, n, seed=42):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, update, 73129]))
    applied = bool(rng.random() < .5)
    lam = float(rng.beta(.2,.2)) if applied else 1.
    permutation = rng.permutation(n).tolist() if applied else list(range(n))
    return dict(applied=applied, lam=lam, permutation=permutation)

def classification_sums(logits, target, partner=None, lam=1.):
    smooth = F.cross_entropy(logits, target, reduction="sum", label_smoothing=.05)
    ordinary = F.cross_entropy(logits.detach(), target, reduction="sum")
    correct = (logits.argmax(1) == target).sum().double()
    if partner is not None:
        smooth = lam*smooth+(1-lam)*F.cross_entropy(logits,partner,reduction="sum",label_smoothing=.05)
        ordinary = lam*ordinary+(1-lam)*F.cross_entropy(logits.detach(),partner,reduction="sum")
        correct = lam*correct+(1-lam)*(logits.argmax(1)==partner).sum().double()
    return smooth, ordinary, correct

def schedule(optimizer, epoch):
    factor = epoch/5 if epoch <= 5 else .5*(1+math.cos(math.pi*(epoch-5)/195))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"]*factor
    return factor

def balanced_indices(global_indices, rank, world):
    result = global_indices[rank::world]
    if not len(result):
        raise ValueError("Empty local batch: choose a world size compatible with the frozen tail.")
    return result

def ddp_backward_loss(local_loss_sum, actual_global_n, world):
    # DDP averages rank gradients: use world/globalN, including the real tail.
    return local_loss_sum*(world/actual_global_n)
