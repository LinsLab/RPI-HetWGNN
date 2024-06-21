import csv
import os

from features.kmer import get_rna_kmer_fea, get_pro_kmer_fea
from features.motif import get_rna_motif_fea, get_pro_motif_fea
from features.pc import get_rna_pc_fea, get_pro_pc_fea
from features.ss import get_rna_ss_fea, get_pro_ss_fea
from utils import read_fasta
import numpy as np

def read_fea_file(fea_file):
    """read features"""
    fea_dict = {}
    f = open(fea_file, 'r')
    for line in f.readlines():
        line = line.strip()
        if len(line.split()) < 5:
            continue
        fea_dict[line.split()[0]] = [float(x) for x in line.split()[1:]]
    f.close()

    return fea_dict

def get_features_files(dataset, rna_seq_file, pro_seq_file):
    features_dir = "./data/"+ dataset +"/"
    rna_ids, rna_seqs = read_fasta(rna_seq_file)
    pro_ids, pro_seqs = read_fasta(pro_seq_file)

    # if not os.path.exists(features_dir+"ncRNA_kmer_features"):
    #     get_rna_kmer_fea(rna_ids, rna_seqs, 4, features_dir, freq=True)
    if not os.path.exists(features_dir + "ncRNA_motif_features"):
        get_rna_motif_fea(rna_ids, rna_seqs, features_dir)
    # if not os.path.exists(features_dir + "ncRNA_pc_features"):
    #     get_rna_pc_fea(rna_ids, rna_seqs, features_dir, fourier_len=10)

    # if not os.path.exists(features_dir+"protein_kmer_features"):
    #     get_pro_kmer_fea(pro_ids, pro_seqs, 3, features_dir, freq=True)
    if not os.path.exists(features_dir + "protein_motif_features"):
        get_pro_motif_fea(pro_ids, pro_seqs, features_dir)
    # if not os.path.exists(features_dir + "protein_pc_features"):
    #     get_pro_pc_fea(pro_ids, pro_seqs, features_dir, fourier_len=10)



def get_pro_features(dataset):
    features_dir = "./data/" + dataset + "/"
    pro_fea = {}
    rna_file = features_dir + 'ncRNA_washed.fa'
    pro_file = features_dir + 'protein_washed.fa'
    get_features_files(dataset, rna_file, pro_file)
    # pro_kmer_fea_file = features_dir + "protein_kmer_features"
    pro_motif_fea_file = features_dir + "protein_motif_features"
    # pro_pc_fea_file = features_dir + "protein_pc_features"
    # pro_kmer_fea = read_fea_file(pro_kmer_fea_file)     # n*343
    pro_motif_fea = read_fea_file(pro_motif_fea_file)  # n*11
    # pro_pc_fea = read_fea_file(pro_pc_fea_file)  # n*80

    pro_ids, pro_seqs = read_fasta(pro_file)
    for r in pro_ids:
        fea_r = pro_motif_fea[r]   # pro_kmer_fea[r]  +  pro_pc_fea[r]
        pro_fea[r] = fea_r
    print("Extract the features of protein has all finished ! ")
    return pro_fea  #11

def get_rna_features(dataset):
    features_dir = "./data/" + dataset + "/"
    rna_fea={}
    rna_file=features_dir+'ncRNA_washed.fa'
    pro_file=features_dir+'protein_washed.fa'
    get_features_files(dataset,rna_file,pro_file)
    # ncRNA_kmer_fea_file = features_dir+"ncRNA_kmer_features"
    ncRNA_motif_fea_file = features_dir+ "ncRNA_motif_features"
    # ncRNA_pc_fea_file = features_dir+ "ncRNA_pc_features"
    # ncRNA_kmer_fea = read_fea_file(ncRNA_kmer_fea_file)    #n*256
    ncRNA_motif_fea = read_fea_file(ncRNA_motif_fea_file)  #n*18
    # ncRNA_pc_fea = read_fea_file(ncRNA_pc_fea_file)         #n*20

    rna_ids, rna_seqs = read_fasta(rna_file)
    for r in rna_ids:
        fea_r =  ncRNA_motif_fea[r]  # ncRNA_kmer_fea[r] +  ncRNA_pc_fea[r]
        rna_fea[r]=fea_r
    print("Extract the features of ncRNA has all finished ! ")
    return rna_fea #274


