#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM


"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k, d):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # if d == 1:
    #     zhishu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    #
    # else:
    #     zhishu = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,41,43,47,53,59,61,67]

    idx = pairwise_distance.topk(k=d * k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx[:, :, ::d]


def get_graph_feature(x, k=20, d=1, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k, d=d)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k, d=d)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # a = (F.pairwise_distance(x.contiguous().view(-1, num_dims), feature.contiguous().view(-1, num_dims), p=2)).view(
    #     batch_size, num_points, k, -1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
    # def __init__(self, args, output_channels=2):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.bn4_2 = nn.BatchNorm2d(64)

        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4_1 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                     self.bn4_1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4_2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                     self.bn4_2,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(64 * 6, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k, d=1)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, d=2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = x1 + x2
        x = get_graph_feature(x, k=self.k, d=4)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = x2 + x3
        x = get_graph_feature(x, k=self.k, d=1)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = x3 + x4
        x = get_graph_feature(x, k=self.k, d=2)
        x = self.conv4_1(x)
        x5 = x.max(dim=-1, keepdim=False)[0]

        x = x4 + x5
        x = get_graph_feature(x, k=self.k, d=4)
        x = self.conv4_2(x)
        x6 = x.max(dim=-1, keepdim=False)[0]
       
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x

'''
class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x    
'''

class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        # self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.bn5_2 = nn.BatchNorm2d(64)
        self.bn5_3 = nn.BatchNorm2d(64)
        self.bn5_4 = nn.BatchNorm2d(64)
        self.bn5_5 = nn.BatchNorm2d(64)
        self.bn5_6 = nn.BatchNorm2d(64)

        self.bn5_7 = nn.BatchNorm2d(64)
        self.bn5_8 = nn.BatchNorm2d(64)
        self.bn5_9 = nn.BatchNorm2d(64)
        self.bn5_10 = nn.BatchNorm2d(64)
        self.bn5_11 = nn.BatchNorm2d(64)
        self.bn5_12 = nn.BatchNorm2d(64)
        self.bn5_13 = nn.BatchNorm2d(64)


        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn5_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5_2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn5_3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_4 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5_4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn5_5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_6 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5_6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5_7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_7,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_8 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                     self.bn5_8,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_9 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_9,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_10 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                     self.bn5_10,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_11,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_12 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                     self.bn5_12,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_13 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_13,
                                     nn.LeakyReLU(negative_slope=0.2))
       

        self.conv6 = nn.Sequential(nn.Conv1d(64*21, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1088+64*21-512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)
        '''
        x0 = get_graph_feature(x, k=self.k, d=1)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        '''
        x = get_graph_feature(x, k=self.k, d=1)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, d=2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1+x2, k=self.k, d=5)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2+x3, k=self.k, d=1)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv5_2(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x3+x4, k=self.k, d=2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_4(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_5(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x5 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x4+x5, k=self.k, d=5)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_6(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_7(x) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x6 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x5 + x6, k=self.k, d=1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_8(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_9(x) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x7 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x6 + x7, k=self.k, d=2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_10(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_11(x) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x8 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x7 + x8, k=self.k, d=5  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_12(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_13(x) # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x9 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

       
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=1)  
        x = self.conv8(x)  
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.bn5_2 = nn.BatchNorm2d(64)
        self.bn5_3 = nn.BatchNorm2d(64)
        self.bn5_4 = nn.BatchNorm2d(64)
        self.bn5_5 = nn.BatchNorm2d(64)
        self.bn5_6 = nn.BatchNorm2d(64)
        self.bn5_7 = nn.BatchNorm2d(64)

        self.bn5_8 = nn.BatchNorm2d(64)
        self.bn5_9 = nn.BatchNorm2d(64)
        self.bn5_10 = nn.BatchNorm2d(64)
        self.bn5_11 = nn.BatchNorm2d(64)
        self.bn5_12 = nn.BatchNorm2d(64)
        self.bn5_13 = nn.BatchNorm2d(64)

        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn5_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn5_2,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_3,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn5_4,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_5,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_6 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn5_6,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_7,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_8 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn5_8,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_9 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_9,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_10 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn5_10,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     self.bn5_11,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_12 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn5_12,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_13 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                      self.bn5_13,
                                      nn.LeakyReLU(negative_slope=0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(64*9, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(args.emb_dims + 64*9, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, d=7, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, d=8)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1+x2, k=self.k, d=9)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_1(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2+x3, k=self.k, d=7)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_3(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x3+x4, k=self.k, d=8)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_4(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_5(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x5 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x4 + x5, k=self.k, d=9)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_6(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_7(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x6 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x5 + x6, k=self.k, d=7)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_8(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_9(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x7 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x6 + x7, k=self.k, d=8)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_10(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_11(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x8 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x7 + x8, k=self.k, d=9)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5_12(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5_13(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x9 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=1)  # (batch_size, 1024+64*9, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*9 num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x