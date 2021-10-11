# -*- coding: utf-8 -*-
import torch
import torch.nn


# original 2D GC loss with no approximation
class GC_2D_Original(torch.nn.Module):

    def __init__(self, lmda, sigma):
        super(GC_2D_Original, self).__init__()
        self.lmda = lmda
        self.sigma = sigma

    def forward(self, input, target):
        # input: B * C * H * W, after sigmoid operation
        # target: B * C * H * W

        # region term equals to BCE
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary_term
        '''
        x5 x1 x6
        x2 x  x4
        x7 x3 x8
        '''
        # vertical: x <-> x1, x3 <-> x1
        target_vert = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # delta(yu, yv)
        input_vert = input[:, :, 1:, :] - input[:, :, :-1, :]  # pu - pv

        # horizontal: x <-> x2, x4 <-> x
        target_hori = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # delta(yu, yv)
        input_hori = input[:, :, :, 1:] - input[:, :, :, :-1]  # pu - pv

        # diagonal1: x <-> x5, x8 <-> x
        target_diag1 = torch.abs(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])  # delta(yu, yv)
        input_diag1 = input[:, :, 1:, 1:] - input[:, :, :-1, :-1]  # pu - pv

        # diagonal2: x <-> x7, x6 <-> x
        target_diag2 = torch.abs(target[:, :, 1:, :-1] - target[:, :, :-1, 1:])  # delta(yu, yv)
        input_diag2 = input[:, :, 1:, :-1] - input[:, :, :-1, 1:]  # pu - pv

        dist1 = 1.0  # dist(u, v), e.g. x <-> x1
        dist2 = 2.0 ** 0.5  # dist(u, v) , e.g. x <-> x6

        p1 = torch.exp(-(input_vert ** 2) / (2 * self.sigma * self.sigma)) / dist1 * target_vert
        p2 = torch.exp(-(input_hori ** 2) / (2 * self.sigma * self.sigma)) / dist1 * target_hori

        p3 = torch.exp(-(input_diag1 ** 2) / (2 * self.sigma * self.sigma)) / dist2 * target_diag1
        p4 = torch.exp(-(input_diag2 ** 2) / (2 * self.sigma * self.sigma)) / dist2 * target_diag2

        boundary_term = (torch.sum(p1) / torch.sum(target_vert) +
                         torch.sum(p2) / torch.sum(target_hori) +
                         torch.sum(p3) / torch.sum(target_diag1) +
                         torch.sum(p4) / torch.sum(target_diag2)) / 4  # equation (5)

        return self.lmda * region_term + boundary_term


# 2D GC loss with boundary approximation in equation (7) to eliminate sigma
class GC_2D(torch.nn.Module):

    def __init__(self, lmda):
        super(GC_2D, self).__init__()
        self.lmda = lmda

    def forward(self, input, target):
        # input: B * C * H * W, after sigmoid operation
        # target: B * C * H * W

        # region term equals to BCE
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary_term
        '''
        x5 x1 x6
        x2 x  x4
        x7 x3 x8
        '''
        # vertical: x <-> x1, x3 <-> x1
        target_vert = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # delta(yu, yv)
        input_vert = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])  # |pu - pv|

        # horizontal: x <-> x2, x4 <-> x
        target_hori = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # delta(yu, yv)
        input_hori = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])  # |pu - pv|

        # diagonal1: x <-> x5, x8 <-> x
        target_diag1 = torch.abs(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])  # delta(yu, yv)
        input_diag1 = torch.abs(input[:, :, 1:, 1:] - input[:, :, :-1, :-1])  # |pu - pv|

        # diagonal2: x <-> x7, x6 <-> x
        target_diag2 = torch.abs(target[:, :, 1:, :-1] - target[:, :, :-1, 1:])  # delta(yu, yv)
        input_diag2 = torch.abs(input[:, :, 1:, :-1] - input[:, :, :-1, 1:])  # |pu - pv|

        p1 = input_vert * target_vert
        p2 = input_hori * target_hori
        p3 = input_diag1 * target_diag1
        p4 = input_diag2 * target_diag2

        boundary_term = 1 - (torch.sum(p1) / torch.sum(target_vert) +
                             torch.sum(p2) / torch.sum(target_hori) +
                             torch.sum(p3) / torch.sum(target_diag1) +
                             torch.sum(p4) / torch.sum(target_diag2)) / 4  # equation (7), and normalized to (0,1)

        return self.lmda * region_term + boundary_term


# 3D GC loss with boundary approximation in equation (7) to eliminate sigma
class GC_3D_v1(torch.nn.Module):
    def __init__(self, lmda):
        super(GC_3D_v1, self).__init__()
        self.lmda = lmda

    def forward(self, input, target):
        # input: B * C * H * W * D, after sigmoid operation
        # target: B * C * H * W * D

        # region term
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary term
        '''
        example [[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]]]
        element 14 has 26 neighborhoods, a total of 13 operations
        '''
        # x5 <-> x14, x14 <-> x23
        input_1 = torch.abs(input[..., 1:, :, :] - input[..., :-1, :, :])  # |pu - pv|
        target_1 = torch.abs(target[..., 1:, :, :] - target[..., :-1, :, :])  # delta(yu, yv)
        # x11 <-> x14, x14 <-> x17
        input_2 = torch.abs(input[..., :, 1:, :] - input[..., :, :-1, :])
        target_2 = torch.abs(target[..., :, 1:, :] - target[..., :, :-1, :])
        # x13 <-> x14, x14 <-> x15
        input_3 = torch.abs(input[..., :, :, 1:] - input[..., :, :, :-1])
        target_3 = torch.abs(target[..., :, :, 1:] - target[..., :, :, :-1])
        # x2 <-> x14, x14 <-> x26
        input_4 = torch.abs(input[..., 1:, 1:, :] - input[..., :-1, :-1, :])
        target_4 = torch.abs(target[..., 1:, 1:, :] - target[..., :-1, :-1, :])
        # x8 <-> x14, x14 <-> x20
        input_5 = torch.abs(input[..., 1:, :-1, :] - input[..., :-1, 1:, :])
        target_5 = torch.abs(target[..., 1:, :-1, :] - target[..., :-1, 1:, :])
        # x10 <-> x14, x14 <-> x18
        input_6 = torch.abs(input[..., :, 1:, 1:] - input[..., :, :-1, :-1])
        target_6 = torch.abs(target[..., :, 1:, 1:] - target[..., :, :-1, :-1])
        # x12 <-> x14, x14 <-> x16
        input_7 = torch.abs(input[..., :, 1:, :-1] - input[..., :, :-1, 1:])
        target_7 = torch.abs(target[..., :, 1:, :-1] - target[..., :, :-1, 1:])
        # x6 <-> x14, x14 <-> x22
        input_8 = torch.abs(input[..., 1:, :, :-1] - input[..., :-1, :, 1:])
        target_8 = torch.abs(target[..., 1:, :, :-1] - target[..., :-1, :, 1:])
        # x4 <-> x14, x14 <-> x24
        input_9 = torch.abs(input[..., 1:, :, 1:] - input[..., :-1, :, :-1])
        target_9 = torch.abs(target[..., 1:, :, 1:] - target[..., :-1, :, :-1])
        # x9 <-> x14, x14 <-> x19
        input_10 = torch.abs(input[..., 1:, :-1, :-1] - input[..., :-1, 1:, 1:])
        target_10 = torch.abs(target[..., 1:, :-1, :-1] - target[..., :-1, 1:, 1:])
        # x3 <-> x14, x14 <-> x25
        input_11 = torch.abs(input[..., 1:, 1:, :-1] - input[..., :-1, :-1, 1:])
        target_11 = torch.abs(target[..., 1:, 1:, :-1] - target[..., :-1, :-1, 1:])
        # x1 <-> x14, x14 <-> x27
        input_12 = torch.abs(input[..., :-1, :-1, :-1] - input[..., 1:, 1:, 1:])
        target_12 = torch.abs(target[..., :-1, :-1, :-1] - target[..., 1:, 1:, 1:])
        # x7 <-> x14, x14 <-> x21
        input_13 = torch.abs(input[..., :-1, 1:, :-1] - input[..., 1:, :-1, 1:])
        target_13 = torch.abs(target[..., :-1, 1:, :-1] - target[..., 1:, :-1, 1:])

        p1 = input_1 * target_1
        p2 = input_2 * target_2
        p3 = input_3 * target_3
        p4 = input_4 * target_4
        p5 = input_5 * target_5
        p6 = input_6 * target_6
        p7 = input_7 * target_7
        p8 = input_8 * target_8
        p9 = input_9 * target_9
        p10 = input_10 * target_10
        p11 = input_11 * target_11
        p12 = input_12 * target_12
        p13 = input_13 * target_13

        smooth = 1e-5  # avoid zero division when target is zero
        boundary_term = 1 - (torch.sum(p1) / (torch.sum(target_1) + smooth) +
                             torch.sum(p2) / (torch.sum(target_2) + smooth) +
                             torch.sum(p3) / (torch.sum(target_3) + smooth) +
                             torch.sum(p4) / (torch.sum(target_4) + smooth) +
                             torch.sum(p5) / (torch.sum(target_5) + smooth) +
                             torch.sum(p6) / (torch.sum(target_6) + smooth) +
                             torch.sum(p7) / (torch.sum(target_7) + smooth) +
                             torch.sum(p8) / (torch.sum(target_8) + smooth) +
                             torch.sum(p9) / (torch.sum(target_9) + smooth) +
                             torch.sum(p10) / (torch.sum(target_10) + smooth) +
                             torch.sum(p11) / (torch.sum(target_11) + smooth) +
                             torch.sum(p12) / (torch.sum(target_12) + smooth) +
                             torch.sum(p13) / (torch.sum(target_13) + smooth)) / 13  # equation (5), and normalized to (0,1)

        return self.lmda * region_term + boundary_term


# this 3D version further eliminates the abs operation
class GC_3D_v2(torch.nn.Module):
    def __init__(self, lmda):
        super(GC_3D_v2, self).__init__()
        self.lmda = lmda

    def forward(self, input, target):
        # input: B * C * H * W * D, after sigmoid operation
        # target: B * C * H * W * D

        # region term
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary term
        '''
        example [[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]]]
        element 14 has 26 neighborhoods, a total of 13 operations
        '''
        # x5 <-> x14, x14 <-> x23
        input_1 = input[..., 1:, :, :] - input[..., :-1, :, :]
        target_1 = target[..., 1:, :, :] - target[..., :-1, :, :]
        # x11 <-> x14, x14 <-> x17
        input_2 = input[..., :, 1:, :] - input[..., :, :-1, :]
        target_2 = target[..., :, 1:, :] - target[..., :, :-1, :]
        # x13 <-> x14, x14 <-> x15
        input_3 = input[..., :, :, 1:] - input[..., :, :, :-1]
        target_3 = target[..., :, :, 1:] - target[..., :, :, :-1]
        # x2 <-> x14, x14 <-> x26
        input_4 = input[..., 1:, 1:, :] - input[..., :-1, :-1, :]
        target_4 = target[..., 1:, 1:, :] - target[..., :-1, :-1, :]
        # x8 <-> x14, x14 <-> x20
        input_5 = input[..., 1:, :-1, :] - input[..., :-1, 1:, :]
        target_5 = target[..., 1:, :-1, :] - target[..., :-1, 1:, :]
        # x10 <-> x14, x14 <-> x18
        input_6 = input[..., :, 1:, 1:] - input[..., :, :-1, :-1]
        target_6 = target[..., :, 1:, 1:] - target[..., :, :-1, :-1]
        # x12 <-> x14, x14 <-> x16
        input_7 = input[..., :, 1:, :-1] - input[..., :, :-1, 1:]
        target_7 = target[..., :, 1:, :-1] - target[..., :, :-1, 1:]
        # x6 <-> x14, x14 <-> x22
        input_8 = input[..., 1:, :, :-1] - input[..., :-1, :, 1:]
        target_8 = target[..., 1:, :, :-1] - target[..., :-1, :, 1:]
        # x4 <-> x14, x14 <-> x24
        input_9 = input[..., 1:, :, 1:] - input[..., :-1, :, :-1]
        target_9 = target[..., 1:, :, 1:] - target[..., :-1, :, :-1]
        # x9 <-> x14, x14 <-> x19
        input_10 = input[..., 1:, :-1, :-1] - input[..., :-1, 1:, 1:]
        target_10 = target[..., 1:, :-1, :-1] - target[..., :-1, 1:, 1:]
        # x3 <-> x14, x14 <-> x25
        input_11 = input[..., 1:, 1:, :-1] - input[..., :-1, :-1, 1:]
        target_11 = target[..., 1:, 1:, :-1] - target[..., :-1, :-1, 1:]
        # x1 <-> x14, x14 <-> x27
        input_12 = input[..., :-1, :-1, :-1] - input[..., 1:, 1:, 1:]
        target_12 = target[..., :-1, :-1, :-1] - target[..., 1:, 1:, 1:]
        # x7 <-> x14, x14 <-> x21
        input_13 = input[..., :-1, 1:, :-1] - input[..., 1:, :-1, 1:]
        target_13 = target[..., :-1, 1:, :-1] - target[..., 1:, :-1, 1:]

        p1 = input_1 * target_1
        p2 = input_2 * target_2
        p3 = input_3 * target_3
        p4 = input_4 * target_4
        p5 = input_5 * target_5
        p6 = input_6 * target_6
        p7 = input_7 * target_7
        p8 = input_8 * target_8
        p9 = input_9 * target_9
        p10 = input_10 * target_10
        p11 = input_11 * target_11
        p12 = input_12 * target_12
        p13 = input_13 * target_13

        smooth = 1e-5  # avoid zero division when target only has one class
        boundary_term = 1 - (torch.sum(p1) / (torch.sum(target_1 * target_1) + smooth) +
                             torch.sum(p2) / (torch.sum(target_2 * target_2) + smooth) +
                             torch.sum(p3) / (torch.sum(target_3 * target_3) + smooth) +
                             torch.sum(p4) / (torch.sum(target_4 * target_4) + smooth) +
                             torch.sum(p5) / (torch.sum(target_5 * target_5) + smooth) +
                             torch.sum(p6) / (torch.sum(target_6 * target_6) + smooth) +
                             torch.sum(p7) / (torch.sum(target_7 * target_7) + smooth) +
                             torch.sum(p8) / (torch.sum(target_8 * target_8) + smooth) +
                             torch.sum(p9) / (torch.sum(target_9 * target_9) + smooth) +
                             torch.sum(p10) / (torch.sum(target_10 * target_10) + smooth) +
                             torch.sum(p11) / (torch.sum(target_11 * target_11) + smooth) +
                             torch.sum(p12) / (torch.sum(target_12 * target_12) + smooth) +
                             torch.sum(p13) / (torch.sum(target_13 * target_13) + smooth)) / 13

        return self.lmda * region_term + boundary_term
