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

human_biogrid_file = 'data/human_ppi/BIOGRID-ORGANISM-Homo_sapiens-4.1.190.tab3.txt'
covid_biogrid_file = 'data/coronavirus_ppi/BIOGRID-CORONAVIRUS-4.1.190.tab3.txt'
uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
covid_fasta_name='data/coronavirus_ppi/covid_proteins.fasta'
# blast_file_name = 'data/coronavirus_ppi/blastp_output.txt'
blast_file_name = 'data/coronavirus_ppi/blastp_output_evalues.txt'
SARSCOV2_ORG_ID='2697049'
HUMAN_ORG_ID='9606'

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
            virus_to_blast[virus_protein][blast_protein] = e_value

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
                        any_to_human[prot_a_sp_id][prot_b_sp_id] = 1
                    if prot_a_org == HUMAN_ORG_ID:
                        any_to_human[prot_b_sp_id][prot_a_sp_id] = 1
    return any_to_human


def get_virus_to_human_preds(virus_to_blast,any_to_human):
    """
    Maps virus protein to a list of its human predictions
    """
    virus_to_human_preds = {}
    for virus_protein,blast_results in virus_to_blast.items():
        virus_to_human_preds[virus_protein] = {}
        for blast_protein in blast_results:
            if blast_protein in any_to_human:
                human_interactions = any_to_human[blast_protein]
                for human_protein in human_interactions: 
                    virus_to_human_preds[virus_protein][human_protein] = blast_results[blast_protein]

    return virus_to_human_preds


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

def compare_pred_target(virus_to_human_preds,virus_to_human_targets):
    """
    Helper function to compare number of predictions and targets for each virus protein
    """
    total_pred_vals = 0
    total_target_vals = 0
    for key in virus_to_human_preds:
        pred_key_len = len(virus_to_human_preds[key])
        if key in virus_to_human_targets:
            target_key_len = len(virus_to_human_targets[key])
        else:
            target_key_len = 0
        total_pred_vals += pred_key_len
        total_target_vals += target_key_len
        print(key,pred_key_len,target_key_len)
    print('Total Vals:',total_pred_vals,total_target_vals)


def evaluate(virus_to_human_preds,virus_to_human_targets):
    precisions = []
    recalls = []
    f1_scores = []

    # stop()

    for virus_protein in virus_to_human_preds:
    # for virus_protein in ['S']:

        if virus_protein in virus_to_human_targets:
            intersections = set(virus_to_human_preds[virus_protein]).intersection(set(virus_to_human_targets[virus_protein]))
            intersections_len = len(intersections)
            # stop()

            if len(virus_to_human_preds[virus_protein]) > 0:
                precision = intersections_len/len(virus_to_human_preds[virus_protein])
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


def main():

    ### Get list of all swissprot proteins
    swissprot_list = fasta_to_dict(uniprot_fasta_name) 

    ### Get all blast results for each virus protein
    virus_to_blast = get_virus_to_blast(blast_file_name)

    ### Get all human interactions for any protein
    any_to_human = get_any_to_human(human_biogrid_file,swissprot_list)

    ### Get human interaction predictions using blast and human interactions
    virus_to_human_preds = get_virus_to_human_preds(virus_to_blast,any_to_human)

    ### Get true virus to human interactions
    virus_to_human_targets = get_virus_to_human_targets(covid_biogrid_file,swissprot_list)

    ### Analysis and Evaluations 
    compare_pred_target(virus_to_human_preds,virus_to_human_targets)
    evaluate(virus_to_human_preds,virus_to_human_targets)

    # summarize_dict(virus_to_blast,'Virus to BLAST')
    # summarize_dict(virus_to_human_preds,'Virus to Human Pred')
    # summarize_dict(virus_to_human_targets,'Virus to Human Target')

    # for key in virus_to_human_targets:
    #     if key not in virus_to_human_preds:
    #         print(key)


if __name__ == "__main__":
    main()