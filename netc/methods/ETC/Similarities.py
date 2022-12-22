import torch
import torch.nn as nn
import torch.nn.functional as F

class SimMatrix(nn.Module):
    def __init__(self, eps=1e-8, act = torch.cos):
        super(SimMatrix, self).__init__()
        self.eps = eps
        self.act = act

    def forward(self, a, b):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=self.eps)
        b_norm = b / torch.clamp(b_n, min=self.eps)
        sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
        return self.act(sim_mt)

class DistMatrix(nn.Module):
    def __init__(self, eps=1e-8):
        super(DistMatrix, self).__init__()
        self.bias = nn.parameter.Parameter(torch.Tensor([1.]))
        self.eps = eps
        
    def forward(self, a, b):
        return ( self.bias + self.eps ) / ( torch.cdist(a, b) + self.bias + self.eps )