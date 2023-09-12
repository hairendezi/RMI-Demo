import torch.nn as nn
from functions import binarize
import math
import torch.nn.functional as F

class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output

class BinaryLinear(nn.Linear):

    def forward(self, input):
        # 二值化参数
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


class BNNModel(nn.Module):
    def __init__(self):
        super(BNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryLinear(64, 100),
            BinaryTanh(),
            BinaryLinear(100, 16)
        )

    def forward(self, x):
        return self.layer1(x)
