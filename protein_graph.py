import torch
import esm
import math
import numpy as np
import json, pickle
import os
from tqdm import tqdm
import os

# data prepare

def protein_graph_construct(proteins, save_dir):
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    protein_graph = {}
    key_list=[]
    for key in proteins:
        key_list.append(key)
    for k_i in tqdm(range(len(key_list))):
        key=key_list[k_i]
        data = []
        pro_id = key
        seq = proteins[key]
        if len(seq) < 1024:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].numpy()
            protein_graph[k_i] = contact_map
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 512
            i = math.ceil(len(seq) / interval)
            for s in range(i):
                start = s * interval   # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub_seq predict end
                # prediction
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # insert into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                if end == len(seq):
                    break
            protein_graph[k_i] = contact_prob_map
        np.save(save_dir + 'pro_' +str(k_i) + '.npy', protein_graph[k_i])

    print('The residues contact maps of proteins have generated successfully!')


def generated_pro_cm(dataset):
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    id=[]
    seq=[]
    proteins_file = './data/'+ dataset +'/protein_washed.fa'
    for line in open(proteins_file, 'r').readlines():
        line=line.strip()
        if line[0] == '>':
            id.append(line[1:])
        else:
            seq.append(line[:])
    proteins=dict(zip(id,seq))

    save_obj(proteins, os.path.join(proteins_file))
    proteins = load_obj(os.path.join(proteins_file))
    save_dir = './data/'+ dataset +'/distance_map/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    protein_graph_construct(proteins,save_dir)
    os.remove(os.path.join(proteins_file + '.pkl'))
    # return protein_graph
