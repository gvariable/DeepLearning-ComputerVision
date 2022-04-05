import math

import tinytorch
from .module import Module
from .. import functional as F
from .. import init
from ..parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(tinytorch.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(tinytorch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameter()

    def reset_parameter(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init.calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(bound, -bound)

    def forward(self, x):
        return F.Linear.apply(x, self.weight, self.bias)
