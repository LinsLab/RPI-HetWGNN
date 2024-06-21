import csv
import numpy as np
import pandas as pd
import math
import random

from tqdm import tqdm

from utils import read_fasta
from Smith_Waterman import *


# np.random.seed(1)
# random.seed(1)
# 计算每对蛋白质之间规范化后的SmithWaterman similarity g1(pi,pj)
def calculate_protein_sw_similarity(pr1, pr2, swscore_matrix):
    score = swscore_matrix[pr1, pr2] / math.sqrt(swscore_matrix[pr1, pr1] * swscore_matrix[pr2, pr2])
    return score


# 计算protein i和ncRNAj之间的互作得分g2
def calculate_socre_of_pri_and_RNAj(pr_i, RNA_j, positive_samples, swscore_matrix):
    score = 0
    related_pair = [pair for pair in positive_samples if pair[0] == RNA_j]
    for pair in related_pair:
        if (pair[1] != pr_i):
            score += calculate_protein_sw_similarity(pr_i, pair[1], swscore_matrix)
    return score


def get_positive_samples(pro, rna, filepath):
    pos = []
    with open(filepath, 'r') as f:
        for pair in f.readlines():
            pair = pair.strip().split()
            if pair[2] == '1':
                pos.append([rna.index(pair[1]), pro.index(pair[0])])
    return pos


def random_match(pos):
    pair=set()
    pos_copy = np.array(pos)
    max_row=np.max(pos_copy[:,0])
    max_col=np.max(pos_copy[:,1])
    while len(pair)<len(pos):
        ran_rna=np.random.randint(0,max_row+1)
        ran_pro=np.random.randint(0,max_col+1)
        if [ran_rna,ran_pro] not in pos:
            pair.add((ran_rna,ran_pro))
    neg=[list(p) for p in pair]
    return neg


def get_Positives_and_Negatives(positive_samples, pr_list, RNA_list, swscore_matrix, savepath):
    Positives = []
    Negatives = []
    ran_neg = random_match(positive_samples)
    for RNA_index in tqdm(range((len(RNA_list)))):
        for pr_index in range(len(pr_list)):
            sample = [RNA_index, pr_index]
            if [RNA_index, pr_index] in positive_samples:
                Ms = 1
                sample.append(Ms)
                Positives.append(sample)
            else:
                Ms = calculate_socre_of_pri_and_RNAj(pr_index, RNA_index, positive_samples, swscore_matrix)
                sample.append(Ms)
                Negatives.append(sample)
                # if sample in ran_neg:
                #     Ms = 0
                #     sample.append(Ms)
                #     Negatives.append(sample)
    Negatives = sorted(Negatives, key=lambda x: x[2])
    Positives = pd.DataFrame(Positives, columns=['RNA', 'protein', 'label'])
    Negatives = pd.DataFrame(Negatives, columns=['RNA', 'protein', 'label'])
    Negatives = Negatives.head(len(Positives))
    Negatives['label'] = 0
    sample=pd.concat([Positives,Negatives],ignore_index=True)
    sample.to_csv(savepath + 'RPI.csv', index=False)

    return sample


def pos_neg_split(dataset):
    savepath = "./data/" + dataset + "/"
    filepath = "./data/" + dataset + '/' + str(dataset) + '.txt'
    rna, _ = read_fasta(savepath + "/ncRNA_washed.fa")
    pro, _ = read_fasta(savepath + "/protein_washed.fa")
    positives = get_positive_samples(pro, rna, filepath)
    if not os.path.exists(savepath + 'protein_sw_mat.csv'):
        generated_SW_matrix(savepath + "protein_washed.fa", savepath)
        print("protein similarity matrix is generated! ")
    swpath = savepath + 'protein_sw_mat.csv'
    swscore_matrix = pd.read_csv(swpath, header=None).values

    sample = get_Positives_and_Negatives(positives, pro, rna, swscore_matrix, savepath)

    return sample

