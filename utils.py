import numpy as np
import torch
import sys
from Bio import SeqIO

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, \
    accuracy_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix


def read_fasta(inputfile):
    try:
        f = open(inputfile, 'r')
    except (IOError,ValueError) as e:
        print(sys.stderr, str(e))
        sys.exit(1)

    seqID=[]
    seqlist=[]
    rescords = list(SeqIO.parse(inputfile, format="fasta"))
    for x in rescords:
        seqID.append(str(x.id))
        seqlist.append(str(x.seq.replace(" ", "")))
    return [seqID,seqlist]

def convert_list_to_str(lists):
    list_str = []
    for l in lists:
        s = ""
        for n in l:
            s += (str(n)+' ')
        list_str.append(s)
    return list_str

def wash_rna_data(rna_fasta_file, out_file):
    ids, Seqs = read_fasta(rna_fasta_file)
    seqs = []
    for seq in Seqs:
        seq = seq.upper()
        seq = seq.replace('T', 'U')
        seqs.append(seq)
    element = ['A', 'C', 'G', 'U']
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] not in element:
                s = seqs[i][j]
                seqs[i] = seqs[i].replace(s, 'A')

    f = open(out_file, 'w')
    for i, s in zip(ids, seqs):
        f.write('>'+i+'\n'+s+'\n')
    f.close()


def wash_pro_data(pro_fasta_file, out_file):
    ids, seqs = read_fasta(pro_fasta_file)

    element = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N', 'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R']
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] not in element:
                s = seqs[i][j]
                # print(s)
                # print(seqs[i])
                seqs[i] = seqs[i].replace(s, 'A')
        while len(seqs[i]) < 15:
            seqs[i] += 'A'
        if len(seqs[i]) >= 10000:
            seqs[i] = seqs[i][:10000]
    f = open(out_file, 'w')
    for i, s in zip(ids, seqs):
        f.write('>'+i+'\n'+s+'\n')
    f.close()

def scaley(y):
    return (y - y.min()) / y.max()

# data write to csv file
def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('ncRNA,protein,label\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')
    f.close()


def show_metrics(label, pred):
    auroc = roc_auc_score(label, pred)
    # precision, recall, prth = precision_recall_curve(label, pred)
    pred = np.round(pred)
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, zero_division=0)
    pre = precision_score(label, pred, zero_division=0)
    mcc = matthews_corrcoef(label, pred)

    conf_matrix = confusion_matrix(label, pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    scores = [auroc*100, acc*100, sen*100, spe*100, f1*100, pre*100, mcc*100]
    return scores