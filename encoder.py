import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv, GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep


class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class WGCN(nn.Module):
    def __init__(self, n_filters=32, embed_dim=128, max_len=3000, num_feature_rna=64, num_feature_pro=33,
                 output_dim=128, dropout=0.2):
        super(WGCN, self).__init__()
        print("WGCN is loading...")
        # ncRNA sequence branch (1D conv)
        self.embedding_xr1 = nn.Embedding(5, embed_dim)
        self.embedding_xr2 = nn.Embedding(num_feature_rna + 1, embed_dim)

        self.conv_xr1 = nn.Conv1d(in_channels = max_len, out_channels=n_filters, kernel_size=8)
        self.conv_xr2 = nn.Conv1d(in_channels = max_len - int(math.log(num_feature_rna, 4)) + 1, out_channels=n_filters,
                                  kernel_size=8)
        self.fc_xr = nn.Linear(n_filters * 121, output_dim)
        # protein graph branch
        self.pro_conv1 = GCNConv(num_feature_pro, num_feature_pro)
        self.norm1 = NodeLevelBatchNorm(num_feature_pro)
        self.pro_conv2 = GCNConv(num_feature_pro, num_feature_pro * 2)
        self.norm2 = NodeLevelBatchNorm(num_feature_pro * 2)
        self.pro_conv3 = GCNConv(num_feature_pro * 2, num_feature_pro * 4)
        self.norm3 = NodeLevelBatchNorm(num_feature_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_feature_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        pro_x, pro_edge_index, pro_weight, pro_batch = data.x, data.edge_index, data.edge_weight, data.batch
        rna_global_fea = data.global_rna
        rna_local_fea = data.local_rna

        xrg = self.embedding_xr1(rna_global_fea)
        xrg = self.conv_xr1(xrg)
        xrg = xrg.view(-1, 32 * 121)
        xrg = self.fc_xr(xrg)

        xrl = self.embedding_xr2(rna_local_fea)
        xrl = self.conv_xr2(xrl)
        xrl = xrl.view(-1, 32 * 121)
        xrl = self.fc_xr(xrl)

        xp = self.norm1(self.pro_conv1(pro_x, pro_edge_index, pro_weight))  #
        xp = self.relu(xp)
        xp = self.norm2(self.pro_conv2(xp, pro_edge_index))
        xp = self.relu(xp)
        xp = self.norm3(self.pro_conv3(xp, pro_edge_index))
        xp = self.relu(xp)
        xp = gap(xp, pro_batch)
        xp = self.relu(self.pro_fc_g1(xp))
        xp = self.dropout(xp)
        xp = self.pro_fc_g2(xp)
        xp = self.dropout(xp)

        xc_rna = torch.mean(torch.stack([xrg, xrl]), dim=0)

        return xc_rna, xp


class WGAT(torch.nn.Module):
    def __init__(self, n_filters=32, embed_dim=128, max_len=3000, num_feature_rna=64, num_feature_pro=33,
                 output_dim=128, dropout=0.2):
        super(WGAT, self).__init__()
        print("WGAT is loading...")
        # ncRNA sequence branch (1D conv)
        self.embedding_xr1 = nn.Embedding(5, embed_dim)
        self.embedding_xr2 = nn.Embedding(num_feature_rna + 1, embed_dim)

        self.conv_xr1 = nn.Conv1d(in_channels = max_len, out_channels=n_filters, kernel_size=8)
        self.conv_xr2 = nn.Conv1d(in_channels = max_len - int(math.log(num_feature_rna, 4)) + 1, out_channels=n_filters,
                                  kernel_size=8)
        self.fc_xr = nn.Linear(n_filters * 121, output_dim)

        self.pro_conv1 = GCNConv(num_feature_pro, num_feature_pro)
        self.pro_conv2 = GATv2Conv(num_feature_pro, num_feature_pro * 2, heads=2, dropout=dropout)
        self.norm1 = NodeLevelBatchNorm(num_feature_pro)
        self.pro_fc_g1 = torch.nn.Linear(num_feature_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        pro_x, pro_edge_index, pro_weight, pro_batch = data.x, data.edge_index, data.edge_weight, data.batch
        rna_global_fea = data.global_rna
        rna_local_fea = data.local_rna

        xrg = self.embedding_xr1(rna_global_fea)
        conv_xr = self.conv_xr1(xrg)
        xrg = conv_xr.view(-1, 32 * 121)
        xrg = self.fc_xr(xrg)

        xrl = self.embedding_xr2(rna_local_fea)
        conv_xr = self.conv_xr2(xrl)
        xrl = conv_xr.view(-1, 32 * 121)
        xrl = self.fc_xr(xrl)

        xp = self.relu(self.pro_conv1(pro_x, pro_edge_index, pro_weight))  #
        xp = self.norm1(xp)
        xp = self.relu(self.pro_conv2(xp, pro_edge_index))
        xp = gap(xp, pro_batch)
        xp = self.relu(self.pro_fc_g1(xp))
        xp = self.dropout(xp)
        xp = self.pro_fc_g2(xp)
        xp = self.dropout(xp)

        xc_rna = torch.mean(torch.stack([xrg, xrl]), dim=0)

        return xc_rna, xp


class WGINE(torch.nn.Module):
    def __init__(self, n_filters=32, embed_dim=128, max_len=3000, num_feature_rna=64, num_feature_pro=33,
                 output_dim=128, dropout=0.2):
        super(WGINE, self).__init__()
        print("WGINE is loading...")
        dim = 16
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # convolution layers
        nn1 = Sequential(Linear(num_feature_pro, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINEConv(nn1, train_eps=True, edge_dim=1)  #
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINEConv(nn2, train_eps=True, edge_dim=1)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINEConv(nn3, train_eps=True, edge_dim=1)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINEConv(nn4, train_eps=True, edge_dim=1)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINEConv(nn5, train_eps=True, edge_dim=1)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xp = Linear(dim, output_dim)

        # 1D convolution on rna sequence
        self.embedding_xr1 = nn.Embedding(5, embed_dim)
        self.embedding_xr2 = nn.Embedding(num_feature_rna + 1, embed_dim)

        self.conv_xr1 = nn.Conv1d(in_channels = max_len, out_channels=n_filters, kernel_size=8)
        self.conv_xr2 = nn.Conv1d(in_channels = max_len - int(math.log(num_feature_rna, 4)) + 1, out_channels=n_filters,
                                  kernel_size=8)
        self.fc_xr = nn.Linear(n_filters * 121, output_dim)

    def forward(self, data):
        pro_x, pro_edge_index, pro_weight, pro_batch = data.x, data.edge_index, data.edge_weight, data.batch
        rna_global_fea = data.global_rna
        rna_local_fea = data.local_rna

        xrg = self.embedding_xr1(rna_global_fea)
        conv_xr = self.conv_xr1(xrg)
        xrg = conv_xr.view(-1, 32 * 121)
        xrg = self.fc_xr(xrg)

        xrl = self.embedding_xr2(rna_local_fea)
        conv_xr = self.conv_xr2(xrl)
        xrl = conv_xr.view(-1, 32 * 121)
        xrl = self.fc_xr(xrl)

        pro_weight = pro_weight.view(-1, 1)
        xp = F.relu(self.conv1(pro_x, pro_edge_index, pro_weight))
        xp = self.bn1(xp)
        xp = F.relu(self.conv2(xp, pro_edge_index, pro_weight))
        xp = self.bn2(xp)
        xp = F.relu(self.conv3(xp, pro_edge_index, pro_weight))
        xp = self.bn3(xp)
        xp = F.relu(self.conv4(xp, pro_edge_index, pro_weight))
        xp = self.bn4(xp)
        xp = F.relu(self.conv5(xp, pro_edge_index, pro_weight))
        xp = self.bn5(xp)
        xp = gap(xp, pro_batch)
        xp = F.relu(self.fc1_xp(xp))
        xp = F.dropout(xp, p=0.2, training=self.training)

        xc_rna = torch.mean(torch.stack([xrg, xrl]), dim=0)

        return xc_rna, xp
