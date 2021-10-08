import json
import numpy as np
import os
from glob import glob
from pdb import set_trace as stop
# from Bio import SeqIO
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

RUN_NUM='1'
random.seed(17) # run 1

# RUN_NUM='2'
# random.seed(27) # run 2

# RUN_NUM='3'
# random.seed(37) # run 3

"""
For Generating covid ppi data from BioGRID
"""
 
human_biogrid_file = 'data/human_ppi/BIOGRID-ORGANISM-Homo_sapiens-4.1.190.tab3.txt'
covid_biogrid_file = 'data/coronavirus_ppi/BIOGRID-CORONAVIRUS-4.1.190.tab3.txt'

uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
covid_fasta_name='data/coronavirus_ppi/covid_proteins.fasta'

blast_file_name = 'data/coronavirus_ppi/blastp_output_evalues.txt'
SARSCOV2_ORG_ID='2697049'
HUMAN_ORG_ID='9606'

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

def fasta_to_dict(fasta_file):
    """
    Dictionary mapping swissprot ids to the sequence
    keys: swissprot_id
    values: sequence
    """
    swissprot_list = {}
    for seq_record in SeqIO.parse(uniprot_fasta_name, "fasta"):
        uniprot_id = str(seq_record.id).split('|')[1]
        sequence = str(seq_record.seq)
        # species = str(seq_record.description).split('OX=')[1].split(' ')[0]
        swissprot_list[uniprot_id] = sequence

    return swissprot_list

def get_virus_to_blast(blast_file):
    """
    Maps virus protein to a list of its blast results
    """
    max_e = 0
    virus_to_blast = {}
    for line in open(blast_file,'r'):
        line = line.strip()
        proteins = line.split(',')
        virus_protein = proteins[0].upper()
        virus_to_blast[virus_protein] = {}

        blast_proteins_withev = proteins[1:]
        for i in range(0,len(blast_proteins_withev),2):
            blast_protein = blast_proteins_withev[i].split('.')[0]
            e_value = float(blast_proteins_withev[i+1])
            # if virus_protein == 'E' and e_value > 1:
            #     stop()
            virus_to_blast[virus_protein][blast_protein] = 11-e_value
            if e_value>max_e: 
                max_e = e_value
            # if blast_protein not in virus_protein: #need this for duplicates
            #     virus_to_blast[virus_protein][blast_protein] = e_value
            # elif e_value < virus_to_blast[virus_protein][blast_protein]:
            #     virus_to_blast[virus_protein][blast_protein] = e_value

    print('MAX',max_e)
    return virus_to_blast

def get_sp_id(swiss_prot_ids,swissprot_list):
    """
    Gets single Swissprot ID from list of IDs if there exists one in the uniprot list
    """
    for sp_id in swiss_prot_ids:
        if sp_id in swissprot_list:
            return sp_id
    return False

def get_sp_ids(line,swissprot_list):
    """
    Gets Swissprot ID and organism for each protein in biogrid line
    """
    prot_a_org = str(line['Organism ID Interactor A'])
    prot_b_org = str(line['Organism ID Interactor B'])
    prot_a_sp_list = line['SWISS-PROT Accessions Interactor A'].split('|')
    prot_b_sp_list = line['SWISS-PROT Accessions Interactor B'].split('|')

    prot_a_sp_id = get_sp_id(prot_a_sp_list,swissprot_list)
    prot_b_sp_id = get_sp_id(prot_b_sp_list,swissprot_list)

    return prot_a_sp_id,prot_a_org,prot_b_sp_id,prot_b_org

def get_any_to_human(human_biogrid_file,swissprot_list):
    """
    Maps any protein to a list of its human interactions
    """
    any_to_human = {}
    with open(human_biogrid_file, 'r') as f:
        for line in csv.DictReader(f,delimiter='\t'):
            prot_a_sp_id,prot_a_org,prot_b_sp_id,prot_b_org = get_sp_ids(line,swissprot_list)
            
            if prot_a_sp_id and prot_b_sp_id:
                if (prot_a_org != SARSCOV2_ORG_ID) and (prot_b_org != SARSCOV2_ORG_ID): #don't include covid
                    if prot_a_sp_id not in any_to_human: any_to_human[prot_a_sp_id] = {}
                    if prot_b_sp_id not in any_to_human: any_to_human[prot_b_sp_id] = {}

                    if prot_b_org == HUMAN_ORG_ID:
                        any_to_human[prot_a_sp_id][prot_b_sp_id] = True
                    if prot_a_org == HUMAN_ORG_ID:
                        any_to_human[prot_b_sp_id][prot_a_sp_id] = True
    return any_to_human


def get_v_h_preds(virus_to_blast,any_to_human):
    """
    Maps virus protein to a list of its human predictions
    """
    v_h_preds = {}
    for virus_protein,blast_results in virus_to_blast.items():
        v_h_preds[virus_protein] = {}
        for blast_protein in blast_results:
            if blast_protein in any_to_human:
                human_interactions = any_to_human[blast_protein]
                for human_protein in human_interactions: 
                    v_h_preds[virus_protein][human_protein] = blast_results[blast_protein]

    return v_h_preds


def get_virus_to_human_targets(covid_biogrid_file,swissprot_list):
    """
    Maps virus protein to a list of its true human interactions
    """
    virus_to_human = {}
    with open(covid_biogrid_file, 'r') as f:
        for line in csv.DictReader(f,delimiter='\t'):
            prot_a_sp_id,prot_a_org,prot_b_sp_id,prot_b_org = get_sp_ids(line,swissprot_list)

            protein_a_symbol = line['Official Symbol Interactor A'].upper()
            protein_b_symbol = line['Official Symbol Interactor B'].upper()
            # if protein_a_symbol == "NSP4" or protein_a_symbol == "NSP5":
            #     print(protein_a_symbol,line['SWISS-PROT Accessions Interactor A'])
            if prot_a_sp_id and prot_b_sp_id:
                if (prot_a_org == SARSCOV2_ORG_ID) and (prot_b_org == HUMAN_ORG_ID):
                    if protein_a_symbol not in virus_to_human: virus_to_human[protein_a_symbol] = {}

                    virus_to_human[protein_a_symbol][prot_b_sp_id] = 1
                elif (prot_b_org == SARSCOV2_ORG_ID) and (prot_a_org == HUMAN_ORG_ID):
                    if protein_b_symbol not in virus_to_human: virus_to_human[protein_b_symbol] = {}

                    virus_to_human[protein_b_symbol][prot_a_sp_id] = 1

    return virus_to_human

def compare_pred_target(v_h_preds,virus_to_human_targets):
    """
    Helper function to compare number of predictions and targets for each virus protein
    """
    total_pred_vals = 0
    total_target_vals = 0
    for key in v_h_preds:
        pred_key_len = len(v_h_preds[key])
        target_key_len = len(virus_to_human_targets[key])
        total_pred_vals += pred_key_len
        total_target_vals += target_key_len
        print(key,pred_key_len,target_key_len)
    print('Total Vals:',total_pred_vals,total_target_vals)


def evaluate(v_h_preds,virus_to_human_targets):
    precisions = []
    recalls = []
    f1_scores = []

    # stop()

    for virus_protein in v_h_preds:

        
        intersections = set(v_h_preds[virus_protein]).intersection(set(virus_to_human_targets[virus_protein]))
        intersections_len = len(intersections)

        if len(v_h_preds[virus_protein]) > 0:
            precision = intersections_len/len(v_h_preds[virus_protein])
            precisions.append(precision)
        else:
            precision = False

        if len(virus_to_human_targets[virus_protein]) > 0:
            recall = intersections_len/len(virus_to_human_targets[virus_protein])
            recalls.append(recall)
        else:
            recall = False

        if precision and recall:
            f1 = (2*precision*recall)/(precision+recall)
            f1_scores.append(f1)
        
            print(virus_protein)
            print('Precision: {:.3f}'.format(precision))
            print('Recall:    {:.3f}'.format(recall))
            print('F1:        {:.3f}'.format(f1))
            print('************')


    precision_mean = np.array(precisions).mean()
    recall_mean = np.array(recalls).mean()
    f1_mean = np.array(f1_scores).mean()

    print('Mean Precision: {:.3f}'.format(precision_mean))
    print('Mean Recall:    {:.3f}'.format(recall_mean))
    print('Mean F1:        {:.3f}'.format(f1_mean))


def summarize_dict(dict_in,dict_name):
    total_blast = 0
    unique_blast = []
    for virus_protein,blast_results in dict_in.items():
        total_blast += len(blast_results)
        for blast_protein in blast_results:
            if blast_protein not in unique_blast: 
                unique_blast.append(blast_protein)

    print('\n'+dict_name)
    print('Total Values:  ',total_blast)
    print('Unique Values: ',len(unique_blast))



def create_uniprot_dict(fasta_file,species=None):
    uniprot_dict = {}
    for seq_record in SeqIO.parse(uniprot_fasta_name, "fasta"):
        uniprot_id = str(seq_record.id).split('|')[1]
        sequence = str(seq_record.seq)
        if species is not None:
            description = str(seq_record.description)
            sample_species = description.split(' OX=')[1].split(' ')[0]
            # gene = description.split('GN=')[1].split(' ')[0]
            if sample_species == species:
                uniprot_dict[uniprot_id] = sequence
        else:
            uniprot_dict[uniprot_id] = sequence
    return uniprot_dict


def create_covid_dict(fasta_file):
    """ My manually processed list of SARS-COV-2 Proteins"""
    covid_dict = {}
    with open(fasta_file,'r') as f:
        seq_string = None
        for line in f:
            line = line.strip()
            if '>' in line:
                if seq_string is not None:
                    covid_dict[protein_name] = seq_string
                protein_name = line.split('>')[1].split(' ')[0].upper()
                if protein_name == 'SPIKE':
                    protein_name = 'S'
                seq_string = ''
            else:
                seq_string+=line
        covid_dict[protein_name] = seq_string
    return covid_dict



def create_v_h_true(virus_protein_dict,human_protein_dict):
    """
    All possible v-h interactions len(virus_protein_dict)*len(human_protein_dict) total
    v_h_true[virus_protein] = {human_protein_sp_id1,human_protein_sp_id2,...}
    """
    v_h_true = {}
    for virus_protein in virus_protein_dict:
        v_h_true[virus_protein] = {}
        for human_protein_sp_id in human_protein_dict:
            v_h_true[virus_protein][human_protein_sp_id] = 0
    return v_h_true

def add_positives_to_v_h_true(covid_biogrid_file,v_h_true,uniprot_dict,virus_protein_dict,human_protein_dict):
    count = 0
    non_found_count = 0
    with open(covid_biogrid_file, 'r') as f:
        d_reader = csv.DictReader(f,delimiter='\t')
        headers = d_reader.fieldnames
        
        for line in d_reader:
            protein_a_organism = line['Organism ID Interactor A']
            protein_b_organism = line['Organism ID Interactor B']
            protein_a_symbol = line['Official Symbol Interactor A'].upper()
            protein_b_symbol = line['Official Symbol Interactor B'].upper()

            protein_a_sp_list = line['SWISS-PROT Accessions Interactor A'].split('|')
            protein_b_sp_list = line['SWISS-PROT Accessions Interactor B'].split('|')

            protein_a_sp_id = get_sp_id(protein_a_sp_list,uniprot_dict)
            protein_b_sp_id = get_sp_id(protein_b_sp_list,uniprot_dict)

            if (protein_a_organism == SARSCOV2_ORG_ID and protein_b_organism == HUMAN_ORG_ID) or (protein_a_organism == HUMAN_ORG_ID and protein_b_organism == SARSCOV2_ORG_ID):
                
                if protein_a_organism == SARSCOV2_ORG_ID:
                    if protein_a_symbol in virus_protein_dict and protein_b_sp_id in human_protein_dict:
                        v_h_true[protein_a_symbol][protein_b_sp_id] = 1
                elif protein_b_organism == SARSCOV2_ORG_ID:
                    if protein_b_symbol in virus_protein_dict and protein_a_sp_id in human_protein_dict:
                        v_h_true[protein_b_symbol][protein_a_sp_id] = 1

                # if protein_a_organism == SARSCOV2_ORG_ID and protein_a_symbol not in virus_protein_dict:
                #     print('V',protein_a_symbol)
                # if protein_b_organism == SARSCOV2_ORG_ID and protein_b_symbol not in virus_protein_dict:
                #     print('V',protein_b_symbol)
                # if protein_a_organism != SARSCOV2_ORG_ID and protein_a_symbol not in human_protein_dict:
                #     print('H',protein_a_symbol)
                # if protein_b_organism != SARSCOV2_ORG_ID and protein_b_symbol not in human_protein_dict:
                #     print('H',protein_b_symbol)
                    

    print('# Not Found: ',non_found_count)
    return v_h_true


def create_interaction_list(v_h_true,virus_protein_dict,unique_human_proteins):
    interaction_list = []
    pos_count = 0
    neg_count = 0
    for virus_protein in v_h_true:
        for human_protein in v_h_true[virus_protein]:
            is_interaction = v_h_true[virus_protein][human_protein]
            if is_interaction == 1:
                pos_count +=1
            else:
                neg_count +=1
            virus_protein_seq = virus_protein_dict[virus_protein]
            human_protein_seq = unique_human_proteins[human_protein]
            
            sample = {}
            sample['protein_1'] = {'id':virus_protein,'primary':virus_protein_seq}
            sample['protein_2'] = {'id':human_protein,'primary':human_protein_seq}
            sample['is_interaction'] = is_interaction
            interaction_list.append(sample)
    
    print('# Positive Pairs: ',pos_count)
    print('# Negative Pairs: ',neg_count)
    print('# Total Pairs   : ',pos_count+neg_count)
    return interaction_list


def create_and_save_spilts(interaction_list,save_root):
    # random.shuffle(interaction_list)
    # train_list = interaction_list[0:int(len(interaction_list)*0.8)]
    # valid_list = interaction_list[int(len(interaction_list)*0.8):int(len(interaction_list)*0.9)]
    # test_list = interaction_list[int(len(interaction_list)*0.9):]

    # with open(save_root+'/all.json', 'w') as fout:
    #     json.dump(interaction_list , fout)
    # with open(save_root+'/train.json', 'w') as fout:
    #     json.dump(train_list , fout)
    # with open(save_root+'/valid.json', 'w') as fout:
    #     json.dump(valid_list , fout)
    # with open(save_root+'/test.json', 'w') as fout:
    #     json.dump(test_list , fout)

    print(save_root+'/all.json')
    with open(save_root+'/all.json', 'w') as fout:
        json.dump(interaction_list , fout)

    # run_bootstrap = False
    # if run_bootstrap:
    #     train_percentages = [0.0,0.1,0.2,0.4,0.6,0.8]
    #     for train_percent in train_percentages:
    #         random.shuffle(interaction_list)
    #         total=len(interaction_list)
    #         train_end = int(total*train_percent)
    #         valid_end = int(total*train_percent) + int(total*0.1)
    #         train_list = interaction_list[0:train_end]
    #         valid_list = interaction_list[train_end:valid_end]
    #         test_list = interaction_list[valid_end:]

    #         with open(save_root+'/train'+str(train_percent).replace('.','')+'_run'+RUN_NUM+'.json', 'w') as fout:
    #             json.dump(train_list , fout)
    #         with open(save_root+'/valid'+str(train_percent).replace('.','')+'_run'+RUN_NUM+'.json', 'w') as fout:
    #             json.dump(valid_list , fout)
    #         with open(save_root+'/test'+str(train_percent).replace('.','')+'_run'+RUN_NUM+'.json', 'w') as fout:
    #             json.dump(test_list , fout)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default='data/coronavirus/', help="")
    args = parser.parse_args()

    print('\n=====> Create uniprot dict')
    swissprot_list = fasta_to_dict(uniprot_fasta_name) 

    print('\n=====> Get covid protein dict')
    virus_protein_dict = create_covid_dict(covid_fasta_name)

    print('\n=====> Get human protein dict')
    human_protein_dict = create_uniprot_dict(uniprot_fasta_name,HUMAN_ORG_ID)

    print('# Human Proteins: ',len(human_protein_dict))
    print('# Covid Proteins: ',len(virus_protein_dict))
    
    print('\n=====> Create V-H interaction graph')
    v_h_true = create_v_h_true(virus_protein_dict,human_protein_dict)

    print('\n=====> Add positive pairs to H-V interaction graph')
    v_h_true = add_positives_to_v_h_true(covid_biogrid_file,v_h_true,swissprot_list,virus_protein_dict,human_protein_dict)


    print('\n=====> Getting BLAST Predictions')
    ### Get all blast results for each virus protein
    virus_to_blast = get_virus_to_blast(blast_file_name)

    ### Get all human interactions for any protein
    any_to_human = get_any_to_human(human_biogrid_file,swissprot_list)

    ### Get human interaction predictions using blast and human interactions
    v_h_preds = get_v_h_preds(virus_to_blast,any_to_human)

    print('\n=====> Getting Results')
    
    aucs = {}
    auprs = {}
    f1s = {}
    precisions ={}
    for virus_protein in v_h_true:
    # for virus_protein in ['S']:
        preds = []
        targs = []
        human_proteins = v_h_true[virus_protein]
        for human_protein in human_proteins:
            true_val = v_h_true[virus_protein][human_protein]
            targs.append(true_val)
            if human_protein in v_h_preds[virus_protein]:
                pred = v_h_preds[virus_protein][human_protein]
                preds.append(pred)
            else:
                preds.append(0)

        preds = np.array(preds)
        targs = np.array(targs)

        sorted_preds,sorted_targs = zip(*sorted(zip(preds,targs),reverse=True))


        f1 = compute_f1_score(targs,np.copy(preds),0)
        f1s[virus_protein] = f1

        precision_100 = compute_precision_score(sorted_targs[0:100],np.copy(sorted_preds[0:100]),0)
        precisions[virus_protein] = precision_100
        # recall_1000 = compute_recall_score(sorted_targs[0:1000],np.copy(sorted_preds[0:1000]),0)

        # fpr, tpr, thresh = roc_curve(targs, preds, pos_label=1)
        # p1_auc = auc(fpr, tpr)
        p1_auc = sklearn.metrics.roc_auc_score(targs, preds, max_fpr=1.00)
        if not math.isnan(p1_auc):
            aucs[virus_protein] = p1_auc
        else:
            aucs[virus_protein] = 0

        precision, recall, thresh = precision_recall_curve(targs, preds)
        p1_aupr = auc(recall,precision) 
        if not math.isnan(p1_aupr):
            auprs[virus_protein] = p1_aupr
        else:
            auprs[virus_protein] = 0

        
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


    f1_vals = list(f1s.values())
    f1_mean = np.array(f1_vals).mean()
    print('\nF1 {:.3f}'.format(f1_mean))
    ordered_dict = OrderedDict(sorted(f1s.items()))
    for key,val in ordered_dict.items(): 
        print("{:7}{:.3f}".format(key+',',val))
    

    precision_vals = list(precisions.values())
    precision_mean = np.array(precision_vals).mean()
    print('\nPrecision {:.3f}'.format(precision_mean))
    ordered_dict = OrderedDict(sorted(precisions.items()))
    for key,val in ordered_dict.items(): 
        print("{:7}{:.3f}".format(key+',',val))

    

    stop()

    

    