import numpy
import scipy.sparse as sp
import scipy
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
import sklearn
from threading import Lock
from threading import Thread
import torch
import math
from pdb import set_trace as stop
import os
# import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
import itertools
import operator
from collections import defaultdict
# !import code; code.interact(local=vars())

# import warnings
# warnings.filterwarnings('always')

def compute_precision_score(all_targets,all_preds_copy,thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    if len(all_preds_copy[all_preds_copy == 1])>0:
        precision = sklearn.metrics.precision_score(all_targets, all_preds_copy)
    else:
        precision = math.nan
    return precision

def compute_recall_score(all_targets,all_preds_copy,thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    if len(all_preds_copy[all_preds_copy == 1])>0:
        recall = sklearn.metrics.recall_score(all_targets, all_preds_copy)
    else:
        recall = math.nan
    return recall


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None):
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()

def compute_precision_at_l5(sequence_lengths, prediction, labels,ignore_index=-1):
    with torch.no_grad():
        valid_mask = labels != ignore_index
        seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
        x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
        valid_mask &= ((y_ind - x_ind) >= 12).unsqueeze(0)
        probs = F.softmax(prediction, 3)[:, :, :, 1]
        valid_mask = valid_mask.type_as(probs)
        correct = 0
        total = 0 
        for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
            masked_prob = (prob * mask).view(-1)
            most_likely = masked_prob.topk(length // 5, sorted=False)
            selected = label.view(-1).gather(0, most_likely.indices)
            correct += selected.sum().float()
            total += selected.numel()
    return correct / total

def spearmanr(target,prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


def compute_f1_score(all_targets,all_preds_copy,thresh):
    all_preds_copy[all_preds_copy >= thresh] = 1
    all_preds_copy[all_preds_copy < thresh] = 0
    f1 = sklearn.metrics.f1_score(all_targets, all_preds_copy)
    return f1

def evaluate(task,all_preds,all_targets,total_loss,total_correct,
             total_preds,all_seq_lens=None,train=True,split_name=None,
             model_name=None,seq1_ids=None,seq2_ids=None):
    mean_loss = total_loss / total_preds
    metrics = {}
    metrics['loss'] = mean_loss

    if task in ['secondary','homology','4prot','solubility','localization','transmembrane','malaria']:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        conf_mat = confusion_matrix(all_targets,all_preds)
        acc = np.diagonal(conf_mat).sum() / conf_mat.sum()
        try:
            f1 = sklearn.metrics.f1_score(all_targets, all_preds)
        except:
            f1=0

        metrics['acc'] = acc
        metrics['f1'] = f1
        return metrics

    elif task in ['ppi']:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        aucs = []
        aucs_25 = []
        auprs = []

        thresh_metrics = {'F1':{},'Precision':{},'Recall':{}}
        THRESHOLDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        for metric in thresh_metrics:
            for threshold in THRESHOLDS:
                thresh_metrics[metric][threshold] = []
                thresh_metrics[metric][threshold] = []
                thresh_metrics[metric][threshold] = []
  
        sorted_preds,sorted_targs = zip(*sorted(zip(all_preds,all_targets),reverse=True))

        num_zero_targets = len(all_targets[all_targets>0])
        num_one_targets = len(all_targets[all_targets==0])
        if num_zero_targets>1 and num_one_targets>1:
            fpr, tpr, thresh = roc_curve(all_targets, all_preds, pos_label=1)
            p1_auc = auc(fpr, tpr)
            if not math.isnan(p1_auc):
                aucs.append(p1_auc)

        precision, recall, thresh = precision_recall_curve(all_targets, all_preds)
        p1_aupr = auc(recall,precision) 
        if not math.isnan(p1_aupr):
            auprs.append(p1_aupr)

        for threshold in THRESHOLDS:
            f1 = compute_f1_score(all_targets,np.copy(all_preds),threshold)
            precision = compute_precision_score(sorted_targs,np.copy(sorted_preds),threshold)
            recall = compute_recall_score(all_targets,np.copy(all_preds),threshold)
            thresh_metrics['F1'][threshold].append(f1)
            thresh_metrics['Precision'][threshold].append(precision)
            thresh_metrics['Recall'][threshold].append(recall)

        metrics['acc'] = np.array(aucs).mean()
        metrics['auc'] = np.array(aucs).mean()
        metrics['aupr'] = np.array(auprs).mean()

        best_f1 = 0
        best_p = 0
        best_r = 0
        for threshold in THRESHOLDS:
            mean_f1 = np.array(thresh_metrics['F1'][threshold]).mean()
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_f1_thresh = str(threshold).replace('.','')
        
            mean_precision = np.array(thresh_metrics['Precision'][threshold]).mean()
            if mean_precision > best_p:
                best_p = mean_precision
                best_p_thresh = str(threshold).replace('.','')
            
            mean_recall = np.array(thresh_metrics['Recall'][threshold]).mean()
            if mean_recall > best_r:
                best_r = mean_recall
                best_r_thresh = str(threshold).replace('.','')
        
        metrics['f1_'+best_f1_thresh] = best_f1
        metrics['precision_'+best_p_thresh] = best_p
        metrics['recall_'+best_r_thresh] = best_r

        return metrics

    elif task in ['biogrid']:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # sorted_s1_ids,sorted_s2_ids,sorted_preds,sorted_targets = zip(*sorted(zip(seq1_ids,seq2_ids,all_preds,all_targets),reverse=True))

        zipped_samples = zip(seq1_ids,seq2_ids,all_preds,all_targets)


        grouped_samples = defaultdict(list)
        for sample in zipped_samples:
            grouped_samples[sample[0]].append(sample)

        # aucs = []
        # aucs_01 = []
        # aucs_05 = []
        # aucs_25 = []
        aucs_100 = {}
        auprs = {}

        thresh_metrics = {'F1':{},'Precision':{},'Recall':{}}
        THRESHOLDS = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        for metric in thresh_metrics:
            for threshold in THRESHOLDS:
                thresh_metrics[metric][threshold] = {}
                thresh_metrics[metric][threshold] = {}
                thresh_metrics[metric][threshold] = {}
  
        for protein1 in grouped_samples:
            aucs_100[protein1] = 0
            auprs[protein1] = 0
            for threshold in THRESHOLDS:
                thresh_metrics['F1'][threshold][protein1] = 0
                thresh_metrics['Precision'][threshold][protein1] = 0
                thresh_metrics['Recall'][threshold][protein1] = 0

            protein1_preds = [i[2] for i in grouped_samples[protein1]]
            protein1_targets = [i[3] for i in grouped_samples[protein1]]

            protein1_targets = np.array(protein1_targets)
            protein1_preds = np.array(protein1_preds)
            
            sorted_preds,sorted_targs = zip(*sorted(zip(protein1_preds,protein1_targets),reverse=True))

            num_zero_targets = len(protein1_targets[protein1_targets>0])
            num_one_targets = len(protein1_targets[protein1_targets==0])
            if num_zero_targets>0 and num_one_targets>0:

                # auc_01 = sklearn.metrics.roc_auc_score(protein1_targets, protein1_preds, max_fpr=0.01)
                # auc_05 = sklearn.metrics.roc_auc_score(protein1_targets, protein1_preds, max_fpr=0.05)
                # auc_25 = sklearn.metrics.roc_auc_score(protein1_targets, protein1_preds, max_fpr=0.25)
                # if not math.isnan(auc_01): aucs_01.append(auc_01)
                # if not math.isnan(auc_05): aucs_05.append(auc_05)
                # if not math.isnan(auc_25): aucs_25.append(auc_25)
                auc_100 = sklearn.metrics.roc_auc_score(protein1_targets, protein1_preds, max_fpr=1.00)
                if not math.isnan(auc_100): 
                    aucs_100[protein1] = auc_100

                precision, recall, thresh = precision_recall_curve(protein1_targets, protein1_preds)
                p1_aupr = auc(recall,precision) 
                if not math.isnan(p1_aupr):
                    auprs[protein1] = p1_aupr

                for threshold in THRESHOLDS:
                    f1 = compute_f1_score(protein1_targets,np.copy(protein1_preds),threshold)
                    precision = compute_precision_score(sorted_targs[0:100],np.copy(sorted_preds[0:100]),threshold)
                    recall = compute_recall_score(protein1_targets,np.copy(protein1_preds),threshold)
                    thresh_metrics['F1'][threshold][protein1] = f1
                    thresh_metrics['Precision'][threshold][protein1] = precision
                    thresh_metrics['Recall'][threshold][protein1] = recall


        mean_aupr = np.array(list(auprs.values())).mean()
        metrics['acc'] = mean_aupr
        # metrics['auc01'] = np.array(aucs_01).mean()
        # metrics['auc05'] = np.array(aucs_05).mean()
        # metrics['auc25'] = np.array(aucs_25).mean()
        metrics['auc100'] = aucs_100
        metrics['aupr'] = auprs

        best_f1 = 0
        best_mean = 0
        best_thresh = 0
        for threshold in THRESHOLDS:
            f1_vals = list(thresh_metrics['F1'][threshold].values())
            mean_f1 = np.array(f1_vals).mean()
            if mean_f1 > best_mean:
                best_mean = mean_f1 
                best_f1 = thresh_metrics['F1'][threshold]
                best_thresh = str(threshold).replace('.','')
        metrics['f1_'+str(best_thresh)] = best_f1
        
        best_mean = 0
        best_p = 0
        best_thresh = 0
        for threshold in THRESHOLDS:
            precision_vals = list(thresh_metrics['Precision'][threshold].values())
            mean_p1000 = np.array(precision_vals).mean()
            if mean_p1000 > best_mean:
                best_mean = mean_p1000
                best_p = thresh_metrics['Precision'][threshold]
                best_thresh = str(threshold).replace('.','')
        metrics['p100_'+str(best_thresh)] = best_p


        # sorted_preds = 1/(1 + np.exp(-np.array(sorted_preds)))
        # sorted_preds = sorted_preds.tolist()
        # f =  open('ranked_spid_human.txt', 'w')
        # for item in zip(sorted_s1_ids,sorted_s2_ids,sorted_preds): f.write(item[0]+'\t'+item[1]+'\t'+str(item[2])+'\n')
        # f.close()

        # if split_name.lower() == 'test' and model_name is not None:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % acc)
        #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
        #     plt.title('Receiver operating characteristic example')
        #     plt.legend(loc="lower right")
        #     plt.savefig(model_name+'/roc_curve.png')
        #     plt.close()

        return metrics
    
    elif task in ['lm']:

        acc = total_correct/total_preds

        ppl = torch.exp(torch.Tensor([mean_loss])).item()
        metrics['acc'] = acc
        metrics['ppl'] = ppl

        return metrics

    elif task in ['fluorescence','stability']:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        acc = spearmanr(all_targets,all_preds)
        metrics['acc'] = acc

        return metrics

    elif task == 'contact':
        if not train:
            all_preds = pad_sequences(all_preds,constant_value=0)
            all_targets = pad_sequences(all_targets,constant_value=-1)

            all_preds = all_preds.reshape(-1,all_preds.shape[2],all_preds.shape[3],all_preds.shape[4])
            all_targets = all_targets.reshape(-1,all_targets.shape[2],all_targets.shape[3])

            all_seq_lens = np.array(all_seq_lens).reshape(-1)
            
            all_preds = torch.tensor(all_preds)
            all_targets = torch.tensor(all_targets)

            all_seq_lens = torch.tensor(all_seq_lens) 

            precision_at_l5 = compute_precision_at_l5(all_seq_lens, all_preds, all_targets).item()

        else:
            precision_at_l5 = 0

        metrics['acc'] = precision_at_l5

        return metrics



    