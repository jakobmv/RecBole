# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
from typing import Tuple, Union
from torch.nn import _reduction as _Reduction


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


# Simplified DESSLoss implementation, keeps everything in a single tensor untill
# it reaches F_dess_loss, where it splits the tensor into mu and sigma.

# Standard PyTorch Loss class, wich all losses inherit from.


class _Loss(nn.Module):
    reduction: str  # Choose to return loss as "mean", "sum" or "none". none means element-wise loss.

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


def get_mu_sigma(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu, sigma = pred.chunk(2, dim=1)  # Split tensor in equal parts.
    return mu, sigma


class DESSLoss(_Loss):
    """DESSLoss, a loss function for training with uncertainty estimates.

    Args:
        pred (torch.Tensor): Shape (batch_dim, 2*emb_dim).
        y (torch.Tensor): Shape (batch_dim, emb_dim, num_targets).
        beta (float): Weight for the uncertainty term. Default: 1.0
        alpha (float): Weight for balancing between single and multi-target loss terms. Default: 0.5
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        beta: float = 1.0,
        alpha: float = 0.5,
        mu_loss = nn.MSELoss,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.beta = beta
        self.alpha = alpha
        self.mu_loss = mu_loss(reduction=reduction)

    def forward(self, pred: torch.Tensor, y: torch.Tensor):
        return F_dess_loss(
            pred, y, beta=self.beta, alpha=self.alpha, reduction=self.reduction, mu_loss=self.mu_loss
        )


def F_dess_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 0.5,
    reduction: str = "mean",
    mu_loss = nn.MSELoss

) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """DESSLoss, a loss function for training with uncertainty estimates.

    Args:
        pred (torch.Tensor): Shape (batch_dim, 2*emb_dim).
        y (torch.Tensor): Shape (batch_dim, emb_dim, num_targets).
        beta (float): Weight for the uncertainty term. Default: 1.0
        alpha (float): Weight for balancing between single and multi-target loss terms. Default: 0.5
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """

    # Keep the single target and multi target criterions in F_dess_loss to seamlessly
    # accomodate a training scenario with alernating single and multi target samples.
    def single_target_criterion(
        mu_pred: torch.Tensor, sigma_pred: torch.Tensor, target: torch.Tensor, moo_loss = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_i = moo_loss(target, mu_pred)
        mu_loss = (d_i) ** 2
        sigma_loss = torch.abs(d_i) - beta * sigma_pred
        return mu_loss, sigma_loss

    def multi_target_criterion(
        mu_pred: torch.Tensor, sigma_pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_mean = torch.mean(target, dim=1)
        d_i = y_mean - mu_pred
        mu_loss = (d_i) ** 2

        delta_Y_i = torch.max(target, dim=1)[0] - torch.min(target, dim=1)[0]
        sigma_loss = (
            alpha * (torch.abs(d_i) - beta * sigma_pred)
            + (1 - alpha) * delta_Y_i * beta * sigma_pred
        )
        return mu_loss, sigma_loss

    mu_pred, sigma_pred = get_mu_sigma(pred)

    # This doesn't work, because we send in a batch at a time

    #print shape of target
    # print(f"Target shape: {target.shape}")

    if len(target.shape) == 3:
        mu_loss, sigma_loss = multi_target_criterion(mu_pred, sigma_pred, target)
    else:
        mu_loss, sigma_loss = single_target_criterion(mu_pred, sigma_pred, target, mu_loss)

    # HERE LOGG mu_loss and sigma_loss in tensorboard
    sigma_loss = torch.abs(sigma_loss)

    combined_loss = torch.cat((mu_loss, sigma_loss), dim=-1)
    if reduction == "none":
        return combined_loss, mu_loss, sigma_loss
    elif reduction == "mean":
        return combined_loss.mean(), mu_loss.mean(), sigma_loss.mean()
    else:  # sum
        return combined_loss.sum(), mu_loss.sum(), sigma_loss.sum()
