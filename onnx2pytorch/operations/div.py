import torch
from torch import nn

# modified
class Div(nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.other = nn.Parameter(torch.tensor(other), requires_grad=False) if other is not None else None

    def forward(self, input, other=None):
        other = other if other is not None else self.other
        res_type = torch.result_type(input, other)
        true_quotient = torch.true_divide(input, other)
        if res_type.is_floating_point:
            res = true_quotient
        else:
            res = torch.floor(true_quotient).to(res_type)
        return res