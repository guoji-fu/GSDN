import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from src import SpGSDNEFConv
from sklearn.metrics import f1_score

import numpy as np
import scipy.sparse as sp
import argparse
import time


class Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hidden_num=256,
                 alpha=0.6,
                 K=4,
                 requires_grad=True):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_num = hidden_num
        self.alpha = alpha
        self.K = K
        self.requires_grad = requires_grad
        self.conv1 = SpGSDNEFConv(self.in_channels, self.hidden_num, self.alpha, self.K)
        self.lin1 = torch.nn.Linear(self.in_channels, self.hidden_num)
        self.conv2 = SpGSDNEFConv(self.hidden_num, self.hidden_num, self.alpha, self.K)
        self.lin2 = torch.nn.Linear(self.hidden_num, self.hidden_num)
        self.conv3 = SpGSDNEFConv(self.hidden_num, self.out_channels, self.alpha, self.K)
        self.lin3 = torch.nn.Linear(self.hidden_num, self.out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


def train(model, optimizer, train_loader, loss_op, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='ppi',                    
                        help='Input graph.')
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
                        default=256,
                        help='Number of hidden units.')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.6,
                        help='a trade-off between feature smoothness and noise.')
    parser.add_argument('--K',
                        type=int,
                        default=4,
                        help='the order of Taylor Series Expansion.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--concat', 
                        type=bool, 
                        default=False,
                        help='Concating all heads or not.')

    args = parser.parse_args()

    return args

def main(args):
    print('-----------SpGSDNEF ppi-----------')
    ## loading data
    dataset = args.input
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_channels=train_dataset.num_features, 
                out_channels=train_dataset.num_classes, 
                hidden_num=args.hidden_num, 
                alpha=args.alpha,
                K=args.K).to(device)
    loss_op = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t1 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, train_loader, loss_op, device)
        val_f1 = test(model, val_loader, device)
        test_f1 = test(model, test_loader, device)
        print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, val_f1, test_f1))
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, val_f1, test_f1))
    t2 = time.time()
    print('training time is: {}'.format(t2-t1))

if __name__ == '__main__':
    main(get_args())
