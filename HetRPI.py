from HGNN import HGNN
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from encoder import WGCN, WGAT, WGINE
from decoder import KAN


class HetRPI(nn.Module):
    def __init__(self, rpi_g, max_len, hidden_dim=128, layers=1, emb_dim=128, output_dim=1):
        super(HetRPI, self).__init__()
        print("HetRPI is loading...")
        self.device = rpi_g.device
        dropout = 0.2
        self.seq_encoder1 = WGCN(embed_dim=emb_dim, max_len=max_len)
        self.seq_encoder2 = WGAT(embed_dim=emb_dim, max_len=max_len)
        self.seq_encoder3 = WGINE(embed_dim=emb_dim, max_len=max_len)
        self.rpi = HGNN(rpi_g, rpi_g.edata['edges'], rpi_g.ndata['nodes'], hidden_dim, num_layer=layers)
        self.rpi_size = self.rpi.get_output_size()
        self.rpi_fc = nn.Sequential(
            nn.Linear(self.rpi_size, self.rpi_size),
            nn.BatchNorm1d(self.rpi_size),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(self.rpi_size, self.rpi_size),
            nn.BatchNorm1d(self.rpi_size),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(self.rpi_size, self.rpi_size),
            nn.BatchNorm1d(self.rpi_size),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.decoder = KAN([self.rpi_size * 2 * 2, 64, output_dim])

    def forward(self, data):
        rpi_emb = self.rpi()
        rpi_emb = self.rpi_fc(rpi_emb)
        rna_rpi_emb = rpi_emb[data[1]]
        pro_rpi_emb = rpi_emb[data[2]]

        seq_emb_rna = (self.seq_encoder1(data[0])[0] + self.seq_encoder2(data[0])[0] + self.seq_encoder3(data[0])[0]) / 3
        seq_emb_pro = (self.seq_encoder1(data[0])[1] + self.seq_encoder2(data[0])[1] + self.seq_encoder3(data[0])[1]) / 3

        # seq_emb_rna = self.seq_encoder1(data[0])[0]
        # seq_emb_pro = self.seq_encoder1(data[0])[1]

        rna_emb = torch.concat([rna_rpi_emb, seq_emb_rna], dim=-1)
        pro_emb = torch.concat([pro_rpi_emb, seq_emb_pro], dim=-1)
        fea = torch.concat([rna_emb, pro_emb], dim=-1)
        return self.decoder(fea)
