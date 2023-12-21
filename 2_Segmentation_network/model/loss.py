import torch.nn as nn
from .dice import DiceLoss
import torch.nn.functional as F


def bce_dice_loss(output, target):
    loss_dice = DiceLoss("binary", from_logits=True)
    loss_bce = nn.BCELoss(size_average=True)
    return loss_bce(output, target) + (1 - loss_dice(output, target))


def dice_loss(output, target):
    loss_fn = DiceLoss("binary", from_logits=True)
    return loss_fn(output, target)


def bce_loss(output, target):
    loss = nn.BCELoss(size_average=True)
    return loss(output, target)


def mse_loss(output, target):
    loss = nn.MSELoss()
    return loss(output, target)

