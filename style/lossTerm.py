#-*-coding:utf-8-*-
"""
- @author: Enigmatisms
- @date: 2021.4.19
Loss Functions defined for Style Transfer Tasks
"""

import torch
from torch import nn

class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = 0.2

    # compute L2 loss for gram matrix x and gram matrix y
    def forward(self, x, y):
        loss = 0.
        for (Nx, Mx, gmx), (Ny, My, gmy) in zip(x, y):
            e_mat = (gmx - gmy) ** 2
            err = torch.sum(e_mat)
            err /= (4. * Nx * Ny * Mx * My)
            loss += self.w * err
        return loss

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return 0.5 * torch.sum((x - y) ** 2)