import torch
from torch.nn import Parameter
from src.inits import glorot, zeros


class DenseGSDNFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha, K, improved=False, bias=True):
        super(DenseGSDNFConv, self).__init__()

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

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=False):
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

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
