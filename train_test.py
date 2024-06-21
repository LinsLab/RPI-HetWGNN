import os
import time
from decimal import Decimal
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import read_fasta, show_metrics


# initialize the dataset
class RPIDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='RPI369', rna=None, pro=None,
                 xr=None, max_len=3000, y=None, pro_id=None, pro_graph=None, transform=None,
                 pre_transform=None):
        super(RPIDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.y = y
        self.process(rna, pro, xr, pro_id, y, pro_graph, max_len)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.datasetID + '_data_rna.pt', self.datasetID + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, rna, pro, xr, pro_id, y, pro_graph, max_len):
        # print(len(xr), len(pro_id), len(y))
        assert (len(xr) == len(pro_id) and len(xr) == len(y)), 'The three lists must have the same length!'
        data_list = []
        data_len = len(xr)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):
            rna_global_fea = xr[i][:max_len]
            rna_local_fea = xr[i][max_len:]
            pro_key = pro_id[i]
            labels = y[i]
            pro_size, pro_features, pro_edge_index, pro_edge_weight = pro_graph[pro_key]

            GCNData = DATA.Data(x=torch.Tensor(pro_features),
                                edge_index=torch.LongTensor(pro_edge_index).transpose(1, 0),
                                edge_weight=torch.FloatTensor(pro_edge_weight),
                                y=torch.FloatTensor([labels]))
            GCNData.global_rna = torch.LongTensor(np.array([rna_global_fea]))
            GCNData.local_rna = torch.LongTensor(np.array([rna_local_fea]))
            GCNData.__setitem__('c_size', torch.LongTensor(pro_size))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.data_list = data_list
        self.rna_key = rna
        self.pro_key = pro

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.data_list[idx], self.rna_key[idx], self.pro_key[idx]


# training function at each epoch
def train(model, device, loss_fn, train_loader, optimizer, epoch, save_file):
    model.to(device).train()
    total_preds = torch.Tensor()
    true_label = torch.Tensor()
    tloss = 0
    for batch_idx, data in enumerate(train_loader):
        data[0].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(torch.Tensor(output).float().to(device), data[0].y.view(-1, 1).float().to(device))
        tloss += loss.item()
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(output)
        total_preds = torch.cat((total_preds, pred.cpu()), 0)
        true_label = torch.cat((true_label, data[0].y.view(-1, 1).cpu()), 0)

    tlabel = true_label.detach().numpy().flatten()
    prob = total_preds.detach().numpy().flatten()
    scores = show_metrics(tlabel, prob)
    res = 'Train epoch %.0f : AUC= %.3f | ACC= %.3f | Sen= %.3f | Spe= %.3f | F1_score= %.3f | Precision= %.3f | MCC= %.3f   | train loss: %.6f\n' % (
        epoch, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], scores[6], tloss)
    print(res)
    save_file.write(res)


# predict
def test(model, device, loss_fn, epoch, test_loader, save_file):
    total_preds = torch.Tensor()
    true_label = torch.Tensor()
    tloss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data[0].to(device)
            output = model(data)
            tloss += loss_fn(torch.Tensor(output).float().to(device), data[0].y.view(-1, 1).float().to(device)).item()
            pred = torch.sigmoid(output)
            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            true_label = torch.cat((true_label, data[0].y.view(-1, 1).cpu()), 0)

    true_label = true_label.detach().numpy().flatten()
    pred_prob = total_preds.detach().numpy().flatten()
    scores = show_metrics(true_label, pred_prob)
    res = 'Test epoch  %.0f : AUC= %.3f | ACC= %.3f | Sen= %.3f | Spe= %.3f | F1_score= %.3f | Precision= %.3f | MCC= %.3f   | test loss:  %.6f\n' % (
        epoch, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], scores[6], tloss)
    print(res)
    save_file.write(res)
    return scores


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    rna, pro = [], []
    for data in data_list:
        rna.append(data[1])
        pro.append(data[2])
    return batchA, rna, pro
