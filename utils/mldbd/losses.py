import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_lambda = 10
_gamma = 10


class LossId(nn.Module):
    def __init__(self):
        super(LossId, self).__init__()
        self.fc_1 = nn.Linear(512, 1)
        self.fc_2 = nn.Linear(512, 1)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.ap20 = nn.AvgPool2d((20, 20))
        self.ap320 = nn.AvgPool2d((320, 320))

    def forward(self, fc, fgs):
        assert fc.device == fgs.device
        k = fc.size(0)
        fc_processed = self.relu_1(self.fc_1(self.ap320(fc).squeeze()))
        fgs_processed = self.relu_2(self.fc_2(self.ap20(fgs).squeeze()))
        loss = (fc_processed - fgs_processed).abs().sum() / k
        return loss


loss_id = LossId().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def loss_bd(input_dbd: Tensor, input_boundary: Tensor, target: Tensor):
    return loss_d(input_dbd, target) + _lambda * loss_b(input_boundary, target)


def loss_d(input: Tensor, target: Tensor):
    k = input.size(0)  # batch size
    loss = (input - target).abs().sum() / k
    return loss

def loss_b(input: Tensor, target: Tensor):
    k = input.size(0)  # batch size
    target_canny = np.zeros(input.shape)
    # TODO: Ugly, optimize somehow?
    for i in range(k):
        target_canny[i] = cv.Canny((target[i].cpu().squeeze(0).numpy() * 255).astype(np.uint8), 100, 200)

    canny = torch.from_numpy(target_canny.astype(np.float32) / 255).to(input.device)
    loss = (input - canny).pow(2).sum() / k
    return loss


def loss_cd(input: Tensor, class1: Tensor, target: Tensor):
    # TODO: This also needs the class as input, which is a binary classification (I assumed one number output
    #  indicating the chance of it being a background, i.e. 0 = foreground, 1 = background), but its actually a
    #  2 dimensional tensor? Also what is the classification label for the ith category? 0 and 255 for
    #  foreground and background? and what is a classification probability? which of the 2 values of the vector? is BCE
    loss = loss_d(input, target) + _gamma * loss_c(class1, target)
    return loss


def loss_c(class1: Tensor, target: Tensor):
    k = class1.size(0)  # batch size
    loss = (-target[:, 0, 0] * torch.log(class1.clamp(0.00001, 0.9999)) - (1 - target[:, 0, 0]) * torch.log(1 - class1.clamp(0.00001, 0.9999))).sum() / k
    return loss


def loss_bbd(input: Tensor, input_boundary: Tensor, target: Tensor, fc: Tensor, fgs: Tensor):
    return loss_bd(input, input_boundary, target) + loss_id(fc, fgs)
