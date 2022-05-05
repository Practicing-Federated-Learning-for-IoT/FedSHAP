#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w, fraction):
    w_avg = copy.deepcopy(w[0]) #copy the weights from the first user in the list
    for key in w_avg.keys():
        w_avg[key] *= (fraction[0]/sum(fraction))
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (fraction[i]/sum(fraction))
        # w_avg[key] = torch.div(w_avg[key], len(w)) # this is wrong implementation since datasets can be unbalanced
    return w_avg
