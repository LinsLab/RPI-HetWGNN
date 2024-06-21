import argparse
import time
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from data_process import create_RP_dataset, load_data
from split import pos_neg_split
from utils import *
from pytorchtools import EarlyStopping
from HetRPI import HetRPI
import os
from train_test import train, test, collate


def run(args):
    np.random.seed(917)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device='cpu'
    dataset_dir = './data/' + args.dataset + '/'
    model_dir = './model/'
    result_dir = './result/'
    os.makedirs(model_dir)
    os.makedirs(result_dir)
    if not os.path.exists(result_dir+args.dataset):
        os.makedirs(result_dir+args.dataset)
    wash_rna_data(os.path.join(dataset_dir, "ncRNA_Sequence.txt"), os.path.join(dataset_dir, "ncRNA_washed.fa"))
    wash_pro_data(os.path.join(dataset_dir, "protein_Sequence.txt"), os.path.join(dataset_dir, "protein_washed.fa"))
    _, rnaseq = read_fasta(dataset_dir + 'ncRNA_washed.fa')
    max_len = min(max(len(seq) for seq in rnaseq), 3000)
    if not os.path.exists(dataset_dir + '/RPI.csv'):
        sample = pos_neg_split(args.dataset)
    else:
        sample = pd.read_csv(dataset_dir + '/RPI.csv')
    pairs = np.array(sample)
    labels = pairs[:, 2].reshape(-1, 1)
    rpi_g = load_data(dataset_dir, device)
    skf = StratifiedKFold(n_splits=args.fold_num, shuffle=True)
    fold = 0
    scores = []
    save_file = os.path.join(result_dir + args.dataset + '/test_result' + time.strftime("_%y_%m_%d-%H-%M", time.localtime()) + '.txt')
    save_file = open(file=save_file, mode='w')
    for train_index, val_index in skf.split(pairs, labels):
        train_rna, test_rna = pairs[train_index, 0], pairs[val_index, 0]
        train_pro, test_pro = pairs[train_index, 1], pairs[val_index, 1]
        train_label, test_label = labels[train_index, 0], labels[val_index, 0]
        # avoid single sample during Batch Normalization
        if len(train_rna) % args.batch_size == 1:
            train_rna, train_pro, train_label = train_rna[:-1], train_pro[:-1], train_label[:-1]
        if len(train_rna) % args.batch_size == 1:
            test_rna, test_pro, test_label = test_rna[:-1], test_pro[:-1], test_label[:-1]

        train_set_entries = [train_rna, train_pro, train_label]
        test_set_entries = [test_rna, test_pro, test_label]
        train_data, test_data = create_RP_dataset(args.dataset, train_set_entries, test_set_entries)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
        if not os.path.isdir(result_dir + args.dataset):
            os.makedirs(result_dir + args.dataset)
        model = HetRPI(rpi_g, max_len=max_len)
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        print("NO.{} fold : Training the model for {} samples and predicting for {} samples...\n".format(
            fold, len(train_loader.dataset), len(test_loader.dataset)))
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        best_test_score = None

        for epoch in range(args.epoch):
            train(model, device, loss_fn, train_loader, opt, epoch, save_file)
            # validation
            model.eval()
            with torch.no_grad():
                test_score = test(model, device, loss_fn, epoch, test_loader, save_file)
            test_f1 = test_score[4]
            if epoch > 25:
                early_stopping(test_f1, model)
                if early_stopping.counter == 0:
                    best_test_score = test_score
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    scores.append(best_test_score)
                    model_file_name = os.path.join(model_dir, args.dataset + time.strftime("-%y-%m-%d-", time.localtime()) + "fold" + str(fold) + '.pt')
                    torch.save(model.state_dict(), model_file_name)
                    break
        print("#####" * 30)
        test_result_str = 'No.%.0f fold test set result : AUC= %.3f | ACC= %.3f | Sen= %.3f | Spe= %.3f | F1_score= %.3f | Precision= %.3f | MCC= %.3f\n\n' \
                          % (fold, best_test_score[0], best_test_score[1], best_test_score[2], best_test_score[3],
                             best_test_score[4], best_test_score[5], best_test_score[6])
        # del train_data, test_data, train_loader,test_loader
        print(test_result_str)
        save_file.write(test_result_str)
        fold += 1

    scores = np.array(scores)
    mean_score = scores.mean(axis=0)
    final_result = 'the %.0f-fold cross-validation final result : AUC= %.3f | ACC= %.3f | Sen= %.3f | Spe= %.3f | F1_score= %.3f | Pre= %.3f | MCC= %.3f\n\n' \
                   % (args.fold_num, mean_score[0], mean_score[1], mean_score[2], mean_score[3], mean_score[4],
                      mean_score[5], mean_score[6])
    print(final_result)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
    mean_score[0], mean_score[1], mean_score[2], mean_score[3], mean_score[4], mean_score[5], mean_score[6]))
    save_file.write(final_result)
    save_file.close()



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RPI-HetWGNN')
    ap.add_argument('--epoch', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--dataset', type=str, default='RPI2241', help='[RPI369, RPI2241, RPI7317, NPInter2]')
    args = ap.parse_args()
    run(args)
