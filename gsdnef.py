import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from src import GSDNEFConv
import numpy as np
import argparse
import time

np.random.seed(2020)
# torch.manual_seed(123)


class Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hidden_num,
                 alpha,
                 K):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_num = hidden_num
        self.alpha = alpha
        self.K = K
        self.conv1 = GSDNEFConv(self.in_channels, self.hidden_num, self.alpha, self.K, cached=True)
        self.conv2 = GSDNEFConv(self.hidden_num, self.out_channels, self.alpha, self.K, cached=True)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data, x):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data, x):
    model.eval()
    logits, accs = model(x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='Cora',
                        # default='CiteSeer',
                        # default='Pubmed',                      
                        help='Input graph.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.6,
                        help='a trade-off between feature smoothness and noise')
    parser.add_argument('--K',
                        type=int,
                        default=4,
                        help='the order of Taylor Series Expansion')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_num', 
                        type=int, 
                        default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--noise', 
                        type=bool, 
                        default=True,
                        help='Data contains noise or not.')
    parser.add_argument('--sigma', 
                        type=float, 
                        default=0,
                        help='The std of noise.')

    args = parser.parse_args()

    return args

def main(args):
    print('---------------gsdnef dataset %s alpha: %s K: %s sigma: %s----------------' % (args.input, args.alpha, args.K, args.sigma))
    ## loading data
    dataset = args.input
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    [m, n] = np.shape(data.x)
    if args.noise:
        x = data.x + torch.Tensor(np.random.normal(0, args.sigma, (m, n)))
    else:
        x = data.x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(in_channels=dataset.num_features, 
                      out_channels=dataset.num_classes, 
                      hidden_num=args.hidden_num, 
                      alpha=args.alpha, 
                      K=args.K).to(device), data.to(device)
    x = x.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0

    t1 = time.time()

    for epoch in range(1, args.epochs):
        train(model, optimizer, data, x)
        train_acc, val_acc, tmp_test_acc = test(model, data, x)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    t2 = time.time()
    print('{:.4f}'.format(test_acc))
    # print('training time is: {}'.format(t2-t1))

if __name__ == '__main__':
    main(get_args())
