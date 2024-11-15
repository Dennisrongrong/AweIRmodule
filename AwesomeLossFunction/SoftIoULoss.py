# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/10 21:10
@Auth ： 归去来兮
@File ：DepthWiseConv.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
import torch.nn as nn
import torch
###SoftIoULoss#####
def SoftIoULoss(pred, target):
    # Old One
    pred = torch.sigmoid(pred)
    smooth = 1
    # print("pred.shape: ", pred.shape)
    # print("target.shape: ", target.shape)
    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)
    loss = 1 - loss.mean()
    # loss = (1 - loss).mean()

    return loss
