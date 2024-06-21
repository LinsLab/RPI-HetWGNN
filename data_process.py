import pickle
import dgl
import os
from tqdm import tqdm
from utils import *
from train_test import RPIDataset
from functools import reduce
from protein_graph import generated_pro_cm

# nomarlize
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# one hot encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one hot encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_id, seq):
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)  #
    return seq_feature


# target sequence to target graph
def sequence_to_pgraph(pro_idx, pro_id, pro_seq, distance_dir):
    pro_edge_index = []
    pro_edge_distance = []
    pro_size = len(pro_seq)
    contact_map_file = os.path.join(distance_dir, 'pro_' + str(pro_idx) + '.npy')
    distance_map = np.load(contact_map_file, allow_pickle=True).astype('float32')
    # the neighbor residue should have a edge
    for i in range(pro_size):
        if i + 1 < pro_size:
            distance_map[i, i + 1] = 1
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold

    for i, j in zip(index_row, index_col):
        pro_edge_index.append([i, j])  # edge
        pro_edge_distance.append(distance_map[i, j])  # edge weight
    pro_feature = seq_feature(pro_id, pro_seq)  #

    return pro_size, pro_feature, pro_edge_index, pro_edge_distance


def RNA_seq(rna_seqs, max_len):
    rt = []
    nucleotide = {'A': 1, 'G': 2, 'C': 3, 'U': 4}
    elements = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'G', 'C', 'U']] * 3)
    element_to_idx = dict(zip(elements, [i + 1 for i in range(64)]))
    for seq in rna_seqs:
        seq = seq[:max_len]
        remaining = max_len - max_len // len(seq) * len(seq)
        seq_new = seq * (max_len // len(seq)) + seq[:remaining]
        seq_idx = np.ones(2 * max_len - 2)
        for i in range(2 * max_len - 2):
            if i < max_len:
                seq_idx[i] = nucleotide[seq_new[i]]
            else:
                seq_idx[i] = element_to_idx[seq_new[i - max_len:i - max_len + 3]]
        rt.append(seq_idx)
    return rt


def create_nodes(data_path):
    rna_file = data_path + '/ncRNA_washed.fa'
    pro_file = data_path + 'protein_washed.fa'
    rnaid, _ = read_fasta(rna_file)
    proid, _ = read_fasta(pro_file)
    with open(data_path + '/nodes.csv', 'w') as f:
        for i, n in enumerate(rnaid):
            f.write(str(i) + '\t' + n + '\t' + '0' + '\n')
        for i, n in enumerate(proid):
            f.write(str(i + len(rnaid)) + '\t' + n + '\t' + '1' + '\n')
    f.close()


def load_data(data_path, device):
    if os.path.exists(os.path.join(data_path, 'rpi_data.pkl')):
        with open(os.path.join(data_path, 'rpi_data.pkl'), 'rb') as save_data:
            rpi_g, e_feat = pickle.load(save_data)
    else:
        edges = []
        e_feat = []
        rnaid, _ = read_fasta(data_path + '/ncRNA_washed.fa')
        with open(data_path + '/RPI.csv', 'r') as file:
            # skip the first line
            file.readline()
            for line in file.readlines():
                r, p, l = line.strip().split(',')
                # discriminate ncRNA and protein ID
                [r, p, l] = [int(r), int(p) + len(rnaid), int(l)]
                if l == 1:
                    edges.append([r, p])
                    edges.append([p, r])
                    e_feat.append(l)
                    e_feat.append(l)
        rpi_g = dgl.graph(edges)
        e_feat = np.array(e_feat)
        create_nodes(data_path)
        nodes_list = []
        with open(data_path + '/nodes.csv', 'r') as file:
            for line in file.readlines():
                i, node_name, node_type = line.strip().split('\t')
                nodes_list.append([int(i), int(node_type)])
        rpi_g.ndata.update({'nodes': torch.tensor(nodes_list)})
        rpi_g.edata.update({'edges': torch.from_numpy(e_feat)})
        with open(os.path.join(data_path, 'rpi_data.pkl'), 'wb') as file:
            pickle.dump([rpi_g, e_feat], file)

    return rpi_g.to(device)


def create_RP_dataset(dataset, train_set_entries, test_set_entries):
    dataset_dir = './data/' + dataset + '/'

    rnaid, rnaseq = read_fasta(dataset_dir + 'ncRNA_washed.fa')
    proid, proseq = read_fasta(dataset_dir + 'protein_washed.fa')
    pro_distance_dir = os.path.join(dataset_dir, 'distance_map')
    if not os.path.exists(pro_distance_dir):
        generated_pro_cm(dataset)  # if not exist,generate contact map
    pro_graph_fea = {}
    for i in tqdm(range(len(proid))):  # weighted map
        pg_t = sequence_to_pgraph(i, proid[i], proseq[i], pro_distance_dir)
        pro_graph_fea[proid[i]] = pg_t
    max_len = min(max(len(seq) for seq in rnaseq), 3000)
    train_rna, train_prot_, train_Y = train_set_entries
    train_rna_seqs = []
    for key in train_rna:
        train_rna_seqs.append(rnaseq[key])
    train_rt = RNA_seq(train_rna_seqs, max_len)
    train_prot = []
    for key in train_prot_:
        train_prot.append(proid[key])
    train_prot_ = np.array([p + len(rnaid) for p in train_prot_])
    train_rt, train_prot, train_Y = np.asarray(train_rt), np.asarray(train_prot), np.asarray(train_Y)
    train_dataset = RPIDataset(root='./data', dataset=dataset, rna=train_rna, pro=train_prot_, xr=train_rt,
                               max_len=max_len, y=train_Y, pro_id=train_prot, pro_graph=pro_graph_fea)

    test_rna, test_prot_, test_Y = test_set_entries
    test_rna_seqs = []
    for key in test_rna:
        test_rna_seqs.append(rnaseq[key])
    test_rt = RNA_seq(test_rna_seqs, max_len)
    test_prot = []
    for key in test_prot_:
        test_prot.append(proid[key])
    test_prot_ = np.array([p + len(rnaid) for p in test_prot_])
    test_rt, test_prot, test_Y = np.asarray(test_rt), np.asarray(test_prot), np.asarray(test_Y)
    test_dataset = RPIDataset(root='./data', dataset=dataset, rna=test_rna, pro=test_prot_, xr=test_rt,
                              max_len=max_len, y=test_Y, pro_id=test_prot, pro_graph=pro_graph_fea)
    return train_dataset, test_dataset

