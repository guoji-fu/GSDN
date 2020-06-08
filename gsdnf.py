import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit, CoraFull
import torch_geometric.transforms as T
from src import GSDNFConv

import argparse
import time


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
        self.conv1 = GSDNFConv(self.in_channels, self.hidden_num, self.alpha, self.K, cached=True)
        self.conv2 = GSDNFConv(self.hidden_num, self.out_channels, self.alpha, self.K, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='cora',
                        # default='citeseer',
                        # default='pubmed',                      
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

def load_dataset(dataset):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

    if dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        return data, num_features, num_classes
    elif dataset == 'reddit':
        dataset = Reddit(path)
    elif dataset == 'corafull':
        dataset = CoraFull(path)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data = dataset[0]

    data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes)

    return data, num_features, num_classes

def main(args):
    print('---------------gsdnf %s %s----------------' % (args.alpha, args.K))
    ## loading data
    data, num_features, num_classes = load_dataset(args.input)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(in_channels=num_features, 
                      out_channels=num_classes, 
                      hidden_num=args.hidden_num, 
                      alpha=args.alpha, 
                      K=args.K).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0

    t1 = time.time()

    for epoch in range(1, args.epochs):
        train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    t2 = time.time()
    print('{:.4f}'.format(test_acc))
    print('training time is: {}'.format(t2-t1))

if __name__ == '__main__':
    main(get_args())
