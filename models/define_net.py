# -*- coding: utf-8 -*-
import torch
from models.TBFFNet import TBFFNet
from models.BiSRNet import BiSRNet
from models.TED import TED
from models.SSCDl import SSCDl
from models.SCanNet import SCanNet
from models.SCD import SCDNet


def choose_net(net_name, num_classes):
    if net_name == 'TBFFNet':
        return TBFFNet(3, num_classes=num_classes).cuda()

    elif net_name == 'BiSRNet':
        return BiSRNet(3, num_classes=num_classes).cuda()

    elif net_name == 'TED':
        return TED(3, num_classes=num_classes).cuda()

    elif net_name == 'SCanNet':
        return SCanNet(3, num_classes=num_classes).cuda()

    elif net_name == 'SSCDl':
        return SSCDl(3, num_classes=num_classes).cuda()

    elif net_name == 'SCD':
        device = [0, 1]
        return torch.nn.DataParallel(SCDNet(3, num_classes=num_classes), device_ids=device).cuda()

    else:
        raise NotImplementedError('Designated model name not supported!')
