# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:30:49 2018

@author: akash
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply