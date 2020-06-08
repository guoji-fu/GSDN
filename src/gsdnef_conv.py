import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from src import GSDNFConv
from torch_geometric.utils import softmax, remove_self_loops

from src.inits import glorot, zeros
import scipy.sparse as sp
import numpy as np

class GSDNEFConv(GSDNFConv):
    def __init__(self, in_channels, out_channels, alpha, K, improved=False,
                 cached=False, bias=True, **kwargs):
        super(GSDNEFConv, self).__init__(in_channels, 
                                         out_channels, 
                                         alpha, 
                                         K,
                                         improved=improved,
                                         cached=cached, 
                                         bias=bias)
        self.beta = Parameter(torch.Tensor(1))
        self.beta.data.fill_(0.1)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = self.EdgeDenoising(x, edge_index, edge_weight)
        return super(GSDNEFConv, self).forward(x, edge_index, edge_weight)

    def EdgeDenoising(self, x, edge_index, edge_weight=None):
        x_norm = F.normalize(x, p=2, dim=-1)
        adj_hat = self.beta * torch.matmul(x_norm, x_norm.t())
        del x_norm

        # print(self.beta.cpu().detach().numpy())
        row, col = edge_index

        del edge_index

        if edge_weight is not None:
            adj_hat[row, col] += edge_weight
        else:
            adj_hat[row, col] += 1

        del row, col

        adj_hat[adj_hat < 0] = 0
        edge_index = adj_hat.nonzero().t().contiguous()
        row, col = edge_index
        edge_weight = adj_hat[row, col]
        del adj_hat, row, col

        edge_index = edge_index.to(x.device)
        edge_weight = edge_weight.to(x.device)

        return edge_index, edge_weight
