import torch
from torch.nn import Parameter
import torch.nn.functional as F
from src.inits import glorot, zeros


class DenseGSDNEFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha, K, improved=False, bias=True):
        super(DenseGSDNEFConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.alpha = alpha
        self.K = K

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.beta = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.beta.data.fill_(0.1)

    def forward(self, x, adj, mask=None, add_loop=False):
        x_norm = F.normalize(x, p=2, dim=-1)
        adj = adj + self.beta * torch.matmul(x_norm, x_norm.t())
        adj[adj < 0] = 0

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(0), dtype=torch.long, device=adj.device)
            adj[idx, idx] = 1 if not self.improved else 2

        x = torch.matmul(x, self.weight)
        out = x
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        for _ in range(self.K):
            out = self.alpha * torch.matmul(adj, out) + x
        out = (1 - self.alpha) * out

        if self.bias is not None:
            out = out + self.bias

        # if mask is not None:
        #     out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
