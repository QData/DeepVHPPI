import json
import numpy as np
import os
from glob import glob
from pdb import set_trace as stop
import random
import csv
from Bio import SeqIO
import copy
import argparse
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import math
import sklearn
from collections import OrderedDict
import warnings
# warnings.filterwarnings('always')  

def compute_f1_score(all_targets,all_preds_copy,thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    f1 = sklearn.metrics.f1_score(all_targets, all_preds_copy)
    return f1

def compute_precision_score(all_targets,all_preds_copy,thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    precision = sklearn.metrics.precision_score(all_targets, all_preds_copy)
    return precision

def compute_recall_score(all_targets,all_preds_copy,thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    recall = sklearn.metrics.recall_score(all_targets, all_preds_copy)
    return recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default='/af11/jjl5sw/HVPPI/test/', help="")
    args = parser.parse_args()

    pred_file = '/af11/jjl5sw/HVPPI/PPI_prediction_result.out'
    target_file = '/af11/jjl5sw/HVPPI/Protein_pair.tsv'
    

    true_dict = {}
    print('Create True Dict')
    for line in open(target_file,'r'):
        line = line.strip().split('\t')
        p1 = line[0]
        p2 = line[1]
        is_interaction = line[2]
        if p2 not in true_dict:
            true_dict[p2] = {}
        true_dict[p2][p1] = int(is_interaction)

    print('Create Pred Dict')
    pred_dict = {}
    k=0
    for line in open(pred_file,'r'):
        if k == 0:
            pass
        else:
            line = line.strip().split('\t')
            p1 = line[0]
            p2 = line[1]
            score = line[2]
            if p2 not in pred_dict:
                pred_dict[p2] = {}
            pred_dict[p2][p1] = float(score)
        k+=1

    thresh_metrics = {'F1':{},'Precision':{},'Recall':{}}
    THRESHOLDS = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]
    for threshold in THRESHOLDS:
        thresh_metrics['F1'][threshold] = {}
        thresh_metrics['Precision'][threshold] = {}
        thresh_metrics['Recall'][threshold] = {}


    aucs = {}
    auprs = {}
    # covid_prots = preds['Pro2ID'].unique()
    # print(covid_prots)
    for p2 in pred_dict:
        aucs[p2] = 0
        auprs[p2] = 0
        for threshold in THRESHOLDS:
            thresh_metrics['F1'][threshold][p2] = 0
            thresh_metrics['Precision'][threshold][p2] = 0
            thresh_metrics['Recall'][threshold][p2] = 0
        print(p2)
        # p2_preds = preds.loc[preds['Pro2ID'] == p2_name]

        preds = []
        targets = []
        # for index, row in p2_preds.iterrows():
        for p1 in pred_dict[p2]:
            # p1 = row['Pro1ID']
            # p2 = row['Pro2ID']
            # score = row['Score']
            score = pred_dict[p2][p1]

            true_val = true_dict[p2][p1]

            preds.append(score)
            targets.append(true_val)

        
        targets = np.array(targets)
        preds = np.array(preds)
        preds,targets = zip(*sorted(zip(preds,targets),reverse=True))
        
        p2_auc = sklearn.metrics.roc_auc_score(targets, preds, max_fpr=1.00)
        
        if not math.isnan(p2_auc):
            aucs[p2] = p2_auc

        precision, recall, thresh = precision_recall_curve(targets, preds)
        p2_aupr = auc(recall,precision) 
        if not math.isnan(p2_aupr):
            auprs[p2] = p2_aupr

        for threshold in THRESHOLDS:
            f1 = compute_f1_score(targets,np.copy(preds),threshold)
            precision = compute_precision_score(targets[0:100],np.copy(preds[0:100]),threshold)
            recall = compute_recall_score(targets,np.copy(preds),threshold)
            thresh_metrics['F1'][threshold][p2] = f1
            thresh_metrics['Precision'][threshold][p2] = precision
            thresh_metrics['Recall'][threshold][p2] = recall


    auc_vals = list(aucs.values())
    auc_mean = np.array(auc_vals).mean()
    print('\nAUROC {:.3f}'.format(auc_mean))
    ordered_dict = OrderedDict(sorted(aucs.items()))
    for key,val in ordered_dict.items(): 
        print("{:7}{:.3f}".format(key+',',val))


    aupr_vals = list(auprs.values())
    aupr_mean = np.array(aupr_vals).mean()
    print('\nAUPR {:.3f}'.format(aupr_mean))
    ordered_dict = OrderedDict(sorted(auprs.items()))
    for key,val in ordered_dict.items(): 
        print("{:7}{:.3f}".format(key+',',val))

    best_f1_mean = 0
    for threshold in THRESHOLDS:
        f1_vals = list(thresh_metrics['F1'][threshold].values())
        f1_mean = np.array(f1_vals).mean()
        if f1_mean > best_f1_mean:
            best_f1_mean = f1_mean
            best_f1_thresh = threshold

    print('\nF1 {:.2f}: {:.3f}'.format(best_f1_thresh,best_f1_mean))
    ordered_dict = OrderedDict(sorted(thresh_metrics['F1'][best_f1_thresh].items()))
    for key,val in ordered_dict.items(): 
        print("{:7}{:.3f}".format(key+',',val))

    
    best_precision_mean = 0
    for threshold in THRESHOLDS:
        precision_vals = list(thresh_metrics['Precision'][threshold].values())
        precision_mean = np.array(precision_vals).mean()
        if precision_mean > best_precision_mean:
            best_precision_mean = precision_mean
            best_precision_thresh = threshold

    print('\nPrecision {:.2f}: {:.3f}'.format(best_precision_thresh,best_precision_mean))
    ordered_dict = OrderedDict(sorted(thresh_metrics['Precision'][best_precision_thresh].items()))
    for key,val in ordered_dict.items(): 
        print("{:7}{:.3f}".format(key+',',val))

    
    # best_recall_mean = 0
    # for threshold in THRESHOLDS:
    #     recall_vals = list(thresh_metrics['Recall'][threshold].values())
    #     recall_mean = np.array(recall_vals).mean()
    #     if recall_mean > best_recall_mean:
    #         best_recall_mean = recall_mean
    #         best_recall_thresh = threshold

    # print('\nrecall {:.2f}: {:.3f}'.format(best_recall_thresh,best_recall_mean))
    # ordered_dict = OrderedDict(sorted(thresh_metrics['Recall'][best_recall_thresh].items()))
    # for key,val in ordered_dict.items(): 
    #     print("{:7}{:.3f}".format(key+',',val))
    

    # stop()

    

    

