import torch
from torch import Tensor
import math
from PIL import Image
import numpy as np


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def precision_score_(ground_truth: Tensor, prediction: Tensor) -> float:
    intersect = (prediction * ground_truth).sum()
    total_pixel_pred = (prediction).sum()
    precision = (intersect / total_pixel_pred).mean()
    return precision.item()


def recall_score_(ground_truth: Tensor, prediction: Tensor) -> float:
    intersect = (prediction * ground_truth).sum()
    total_pixel_truth = (ground_truth).sum()
    recall = (intersect / total_pixel_truth).mean()
    return recall.item()


def f_beta_measure(ground_truth: Tensor, prediction: Tensor, beta: float = 0.3) -> float:
    p = precision_score_(ground_truth, prediction)
    r = recall_score_(ground_truth, prediction)
    if p == 0 or math.isnan(p) or math.isnan(r):
        return 0
    else:
        return ((1 + beta ** 2) * p * r) / (beta ** 2 * p + r)


def binary_dice_loss(ground_truth: Tensor, prediction: Tensor, smooth: float = 1, p: float = 2,
                     reduction: str = 'mean') -> Tensor:
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        prediction: A tensor of shape [N, *]
        ground_truth: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    See:
        https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
    """
    assert prediction.shape[0] == ground_truth.shape[0], "predict & target batch size don't match"
    predict = prediction.contiguous().view(prediction.shape[0], -1)
    target = ground_truth.contiguous().view(ground_truth.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

    loss = 1 - num / den

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


def cyclic_lr_scale_fn(x, c1: float = -0.3, c2: float = 10):
    return 1 / (1 + math.e ** (-c1 * (x - c2)))


img_gt = np.asarray(Image.open('../data/cvl/testset/gt/computer/0.4/0170-3_1.tiff')) / 255.0
# img_gt = np.asarray(Image.open('../data/cvl/testset/gt/computer/0.4/0163-6_1.tiff')) / 255.0
# img_gt = np.asarray(Image.open('../data/cvl/testset/gt/computer/0.4/0096-4_0.tiff')) / 255.0
img_pred = np.ones_like(img_gt)
img_pred[0] = 1

print(precision_score_(torch.as_tensor(img_gt.copy()), torch.as_tensor(img_pred.copy())))
print(recall_score_(torch.as_tensor(img_gt.copy()), torch.as_tensor(img_pred.copy())))
print(f_beta_measure(torch.as_tensor(img_gt.copy()), torch.as_tensor(img_pred.copy())))
