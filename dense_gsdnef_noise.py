import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, Reddit
import torch_geometric.transforms as T
from src import DenseGSDNEFConv
import numpy as np
import argparse
import time

np.random.seed(2020)


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
        self.conv1 = DenseGSDNEFConv(self.in_channels, self.hidden_num, self.alpha, self.K)
        self.conv2 = DenseGSDNEFConv(self.hidden_num, self.out_channels, self.alpha, self.K)
    
    def forward(self, x, adj):
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, adj)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data, x, adj):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(x, adj)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data, x, adj):
    model.eval()
    logits, accs = model(x, adj), []
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
                        help='The std of feature noise.')
    parser.add_argument('--ratio', 
                        type=float, 
                        default=0,
                        help='The ratio of edge noise.')

    args = parser.parse_args()

    return args

def generate_split(data, num_classes, seed=2020, train_num_per_c=20, val_num_per_c=30):
    torch.manual_seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask

# def load_data(args):
#     dataset = args.input
#     path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
#     dataset = Planetoid(path, dataset, T.NormalizeFeatures())
#     data = dataset[0]

#     ## Generate feature noise and edge noise
#     [m, n] = np.shape(data.x)
#     if args.noise:
#         ## Generate feature noise
#         x_noisy = data.x + torch.Tensor(np.random.normal(0, args.sigma, (m, n)))

#         ## Generate edge noise
#         row, col = data.edge_index
#         added_row, added_col = torch.randint(m, (2, int(m * args.ratio)), dtype=torch.long)
#         # Find duplicated edges ...
#         idx = torch.cat([m * row + col, m * added_row + added_col], dim=0)
#         _, inv, count = torch.unique(idx, return_inverse=True, return_counts=True)
#         keep_mask = count[inv][:row.size(0)] == 1
#         added_mask = count[inv][row.size(0):] == 1
#         keep_edge_index = torch.stack([row[keep_mask], col[keep_mask]], dim=0)
#         added_edge_index = torch.stack([added_row[added_mask], added_col[added_mask]], dim=0)
#         edge_noisy = torch.cat([keep_edge_index, added_edge_index], dim=-1)
#         adj_noisy = torch.zeros((data.x.size(0), data.x.size(0)))
#         col, row = edge_noisy
#         adj_noisy[col, row] = 1
#     else:
#         x_noisy = data.x
#         adj_noisy = torch.zeros((data.x.size(0), data.x.size(0)))
#         col, row = data.edge_index
#         adj_noisy[col, row] = 1

#     return dataset, data, x_noisy, adj_noisy

def load_data(args):
    dataset = args.input
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

    if dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        return data, num_features, num_classes
    elif dataset == 'corafull':
        dataset = CoraFull(path)
    elif dataset in ['cs', 'physics']:
        dataset = Coauthor(path, name=dataset)
    elif dataset in ['computers', 'photo']:
        dataset = Amazon(path, name=dataset)
    elif dataset == 'reddit':
        dataset = Reddit(path)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        return data, num_features, num_classes
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data = dataset[0]

    data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes)

    return data, num_features, num_classes

def generate_noise(args, data):
    ## Generate feature noise and edge noise
    [m, n] = np.shape(data.x)
    if args.noise:
        ## Generate feature noise
        x_noisy = data.x + torch.Tensor(np.random.normal(0, args.sigma, (m, n)))

        ## Generate edge noise
        row, col = data.edge_index
        added_row, added_col = torch.randint(m, (2, int(m * args.ratio)), dtype=torch.long)
        # Find duplicated edges ...
        idx = torch.cat([m * row + col, m * added_row + added_col], dim=0)
        _, inv, count = torch.unique(idx, return_inverse=True, return_counts=True)
        keep_mask = count[inv][:row.size(0)] == 1
        added_mask = count[inv][row.size(0):] == 1
        keep_edge_index = torch.stack([row[keep_mask], col[keep_mask]], dim=0)
        added_edge_index = torch.stack([added_row[added_mask], added_col[added_mask]], dim=0)
        edge_noisy = torch.cat([keep_edge_index, added_edge_index], dim=-1)
        adj_noisy = torch.zeros((data.x.size(0), data.x.size(0)))
        col, row = edge_noisy
        adj_noisy[col, row] = 1
    else:
        x_noisy = data.x
        adj_noisy = torch.zeros((data.x.size(0), data.x.size(0)))
        col, row = data.edge_index
        adj_noisy[col, row] = 1

    return data, x_noisy, adj_noisy

def run_model(model, args, data, x_noisy, adj_noisy, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    x_noisy = x_noisy.to(device)
    adj_noisy = adj_noisy.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0

    t1 = time.time()

    for epoch in range(1, args.epochs):
        train(model, optimizer, data, x_noisy, adj_noisy)
        train_acc, val_acc, tmp_test_acc = test(model, data, x_noisy, adj_noisy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, train_acc, best_val_acc, test_acc))

    t2 = time.time()
    print('{:.4f}'.format(test_acc))
    # print('training time is: {}'.format(t2-t1))

def main(args):
    data, num_features, num_classes = load_data(args)
    data, x_noisy, adj_noisy = generate_noise(args, data)
    print('---------------dense gsdnef dataset %s alpha: %s K: %s sigma: %s ratio: %s----------------' % (args.input, args.alpha, args.K, args.sigma, args.ratio))
    gsdnef_model = Net(in_channels=num_features, 
                        out_channels=num_classes, 
                        hidden_num=args.hidden_num, 
                        alpha=args.alpha, 
                        K=args.K)
    run_model(gsdnef_model, args, data, x_noisy, adj_noisy, 0.02)

    print('---------------dense gsdnef dataset %s alpha: 1.2 K: %s sigma: %s ratio: %s----------------' % (args.input, args.K, args.sigma, args.ratio))
    gsdnef_model = Net(in_channels=num_features, 
                        out_channels=num_classes, 
                        hidden_num=args.hidden_num, 
                        alpha=1.2, 
                        K=args.K)
    run_model(gsdnef_model, args, data, x_noisy, adj_noisy, 0.02)

if __name__ == '__main__':
    main(get_args())
