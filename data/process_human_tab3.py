import json
import numpy as np
import os
from glob import glob
from pdb import set_trace as stop
from Bio import SeqIO
import random
import csv
import argparse

random.seed(7)

"""
For generating human ppi data from BioGRID
"""

human_biogrid_file = 'data/human_ppi/BIOGRID-ORGANISM-Homo_sapiens-4.1.190.tab3.txt'
coronavirus_biogrid_file = 'data/cornavirus/BIOGRID-CORONAVIRUS-4.1.190.tab3.txt'
uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
SARSCOV2_ORG_ID='2697049'
HUMAN_ORG_ID='9606'

def fasta_to_dict(fasta_file):
    """
    Dictionary mapping swissprot ids to the sequence
    keys: swissprot_id
    values: sequence
    """
    uniprot_dict = {}
    for seq_record in SeqIO.parse(uniprot_fasta_name, "fasta"):
        uniprot_id = str(seq_record.id).split('|')[1]
        sequence = str(seq_record.seq)
        # species = str(seq_record.description).split('OX=')[1].split(' ')[0]
        uniprot_dict[uniprot_id] = sequence

    return uniprot_dict

def get_sp_id(swiss_prot_ids,uniprot_dict):
    for sp_id in swiss_prot_ids:
        if sp_id in uniprot_dict:
            return sp_id,uniprot_dict[sp_id]
    return False,False

def get_sequences(line,uniprot_dict):
    prot_a_org = str(line['Organism ID Interactor A'])
    prot_b_org = str(line['Organism ID Interactor B'])
    prot_a_sp_list = line['SWISS-PROT Accessions Interactor A'].split('|')
    prot_b_sp_list = line['SWISS-PROT Accessions Interactor B'].split('|')

    prot_a_sp_id,prot_a_seq = get_sp_id(prot_a_sp_list,uniprot_dict)
    prot_b_sp_id,prot_b_seq = get_sp_id(prot_b_sp_list,uniprot_dict)

    if prot_a_org != SARSCOV2_ORG_ID and prot_b_org != SARSCOV2_ORG_ID: #don't include in training
        if prot_a_org == HUMAN_ORG_ID or prot_b_org == HUMAN_ORG_ID: # at least 1 human protein
            # if prot_a_org != prot_b_org: #don't include in H-H
            return prot_a_sp_id,prot_a_seq,prot_a_org,prot_b_sp_id,prot_b_seq,prot_b_org
        
    return False,False,False,False,False,False



def get_protein_dict(human_biogrid_file,uniprot_dict):
    """
    Find all unique proteins in the dataset, which we will use as nodes for our graph.
    """
    protein_to_idx = {}
    with open(human_biogrid_file, 'r') as f:
        d_reader = csv.DictReader(f,delimiter='\t')
        headers = d_reader.fieldnames
        for line in d_reader:
            prot_a_sp_id,prot_a_seq,prot_a_org,prot_b_sp_id,prot_b_seq,prot_b_org = get_sequences(line,uniprot_dict)
            if prot_a_seq and prot_b_seq:
                if prot_a_sp_id not in protein_to_idx:
                    protein_to_idx[prot_a_sp_id] = {'idx':len(protein_to_idx),'primary':prot_a_seq,'organism':prot_a_org}
                if prot_b_sp_id not in protein_to_idx:
                    protein_to_idx[prot_b_sp_id] = {'idx':len(protein_to_idx),'primary':prot_b_seq,'organism':prot_b_org}
    
    idx_to_protein = {v['idx']: {'sp_id':k,'primary':v['primary'],'organism':v['organism']} for k, v in protein_to_idx.items()}
    return protein_to_idx,idx_to_protein


def create_interaction_graph(human_biogrid_file,protein_to_idx):
    """
    Create PPI adjacency matrix with all zeros and fill in positive interactions.
    """
    idx_count = 0
    interactions_adj_mat = np.zeros((len(protein_to_idx),len(protein_to_idx)))
    with open(human_biogrid_file, 'r') as f:
        d_reader = csv.DictReader(f,delimiter='\t')
        headers = d_reader.fieldnames
        for line in d_reader:
            prot_a_sp_id,prot_a_seq,prot_a_org,prot_b_sp_id,prot_b_seq,prot_b_org = get_sequences(line,uniprot_dict)
            if prot_a_seq and prot_b_seq:
                idx_a = protein_to_idx[prot_a_sp_id]['idx']
                idx_b = protein_to_idx[prot_b_sp_id]['idx']

                org_a = protein_to_idx[prot_a_sp_id]['organism']
                org_b = protein_to_idx[prot_b_sp_id]['organism']

                # Add to both (a,b) and (b,a) because we don't want directionality
                interactions_adj_mat[idx_a][idx_b] = 1
                interactions_adj_mat[idx_b][idx_a] = 1

    return interactions_adj_mat

def create_sample(prot_a_sp_id,prot_a_seq,prot_b_sp_id,prot_b_seq,is_interaction):
    sample = {}
    sample['protein_1'] = {'id':prot_a_sp_id,'primary':prot_a_seq}
    sample['protein_2'] = {'id':prot_b_sp_id,'primary':prot_b_seq}
    sample['is_interaction'] = is_interaction
    return sample

def get_positive_pairs(args,interactions_adj_mat,uniprot_dict,idx_to_protein):
    """
    Get all nonzero elements from the interactions_adj_mat
    """
    interaction_list = []
    nonzero = np.transpose(np.nonzero(interactions_adj_mat))
    for (idx_a,idx_b) in nonzero:
        if idx_a < idx_b: #only count each pair once
            prot_a_sp_id = idx_to_protein[idx_a]['sp_id']
            prot_a_seq = idx_to_protein[idx_a]['primary']
            prot_a_org = idx_to_protein[idx_a]['organism']
            prot_b_sp_id = idx_to_protein[idx_b]['sp_id']
            prot_b_seq = idx_to_protein[idx_b]['primary']
            prot_b_org = idx_to_protein[idx_b]['organism']

            if args.type in ['hh','hh_hv'] and (prot_a_org == HUMAN_ORG_ID) and (prot_b_org == HUMAN_ORG_ID):
                sample = create_sample(prot_a_sp_id,prot_a_seq,prot_b_sp_id,prot_b_seq,1)
                interaction_list.append(sample)
                sample = create_sample(prot_b_sp_id,prot_b_seq,prot_a_sp_id,prot_a_seq,1)
                interaction_list.append(sample)
            elif args.type in ['hv','hh_hv'] and (prot_a_org != HUMAN_ORG_ID) and (prot_b_org == HUMAN_ORG_ID):
                sample = create_sample(prot_a_sp_id,prot_a_seq,prot_b_sp_id,prot_b_seq,1)
                interaction_list.append(sample)
            elif args.type in ['hv','hh_hv'] and (prot_b_org != HUMAN_ORG_ID) and (prot_a_org == HUMAN_ORG_ID):
                sample = create_sample(prot_b_sp_id,prot_b_seq,prot_a_sp_id,prot_a_seq,1)
                interaction_list.append(sample)


    return interaction_list

def get_negative_pairs(args,interactions_adj_mat,uniprot_dict,idx_to_protein,num_samples):
    """
    Randomly sample num_samples of negative pairs
    """

    interaction_list = []
    zero = np.transpose(np.where(interactions_adj_mat == 0))
    neg_count = 0
    # random sample from full zero array takes too long, so we have to
    # sample from a subset of the zero array
    max_samples = num_samples*100 
    for zero_pair_idx in np.random.choice(len(zero), max_samples):
        (idx_a,idx_b) = zero[zero_pair_idx]
        if idx_a < idx_b:
            prot_a_sp_id = idx_to_protein[idx_a]['sp_id']
            prot_a_seq = idx_to_protein[idx_a]['primary']
            prot_a_org = idx_to_protein[idx_a]['organism']
            prot_b_sp_id = idx_to_protein[idx_b]['sp_id']
            prot_b_seq = idx_to_protein[idx_b]['primary']
            prot_b_org = idx_to_protein[idx_b]['organism']

            # Only use pairs with EXACTLY one human protein (since that will be our test case)
            if args.type in ['hh','hh_hv'] and (prot_a_org == HUMAN_ORG_ID) and (prot_b_org == HUMAN_ORG_ID):
                sample = create_sample(prot_a_sp_id,prot_a_seq,prot_b_sp_id,prot_b_seq,0)
                interaction_list.append(sample)
                neg_count +=1
            elif args.type in ['hv','hh_hv'] and (prot_a_org != HUMAN_ORG_ID) and (prot_b_org == HUMAN_ORG_ID):
                sample = create_sample(prot_b_sp_id,prot_b_seq,prot_a_sp_id,prot_a_seq,0)
                interaction_list.append(sample)
                neg_count +=1
            elif args.type in ['hv','hh_hv'] and (prot_b_org != HUMAN_ORG_ID) and (prot_a_org == HUMAN_ORG_ID):
                sample = create_sample(prot_a_sp_id,prot_a_seq,prot_b_sp_id,prot_b_seq,0)
                interaction_list.append(sample)
                neg_count +=1

            if neg_count == num_samples:
                return interaction_list,neg_count

    return interaction_list,neg_count

def create_and_save_spilts(args,interaction_list,neg_multiplier,save_root):
    random.shuffle(interaction_list)
    train_list = interaction_list[0:int(len(interaction_list)*0.9)]
    valid_list = interaction_list[int(len(interaction_list)*0.9):]
    
    print('# Samples:',len(interaction_list))
    print('# Train  :',len(train_list))
    print('# Valid  :',len(valid_list))


    with open(save_root+'/all'+str(neg_multiplier)+'x_'+args.type+'.json', 'w') as fout:
        json.dump(interaction_list , fout)
    # with open(save_root+'/train'+str(neg_multiplier)+'x_'+args.type+'.json', 'w') as fout:
    #     json.dump(train_list , fout)
    # with open(save_root+'/valid'+str(neg_multiplier)+'x_'+args.type+'.json', 'w') as fout:
    #     json.dump(valid_list , fout)

    return True

def print_protein_counts(protein_to_idx):
    human_prot_count =0
    for key in protein_to_idx: 
        if protein_to_idx[key]['organism'] == HUMAN_ORG_ID: 
            human_prot_count+=1

    print('# Human Proteins: ',human_prot_count)
    print('# Other Proteins: ',len(protein_to_idx)-human_prot_count)
    print('# Total Proteins: ',len(protein_to_idx))

    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--neg_multiplier", type=int, default=1, help="how many times more negative than postive?")
    parser.add_argument('--type', type=str, choices=['hh', 'hv','hh_hv'], default='hh_hv')
    parser.add_argument("--save_root", type=str, default='data/human_ppi/', help="")
    args = parser.parse_args()

    print('\n=====> Create uniprot dict')
    uniprot_dict = fasta_to_dict(uniprot_fasta_name)

    print('\n=====> Get protein_to_idx')
    protein_to_idx,idx_to_protein = get_protein_dict(human_biogrid_file,uniprot_dict)
    print_protein_counts(protein_to_idx)

    print('\n=====> Create interaction graph')
    interactions_adj_mat = create_interaction_graph(human_biogrid_file,protein_to_idx)

    print('\n=====> Get Positive Pairs')
    pos_interaction_list = get_positive_pairs(args,interactions_adj_mat,uniprot_dict,idx_to_protein)
    pos_count = len(pos_interaction_list)
    num_neg = pos_count*args.neg_multiplier
    print('# Positive Pairs:  ',pos_count)
    
    print('\n=====> Get Negative Pairs')
    neg_interaction_list,neg_count = get_negative_pairs(args,interactions_adj_mat,uniprot_dict,idx_to_protein,num_neg)
    assert num_neg == neg_count
    print('# Negative Pairs:  ',neg_count)

    full_interaction_list = pos_interaction_list+neg_interaction_list
    
    print('\n=====> Create and save splits')
    create_and_save_spilts(args,full_interaction_list,args.neg_multiplier,save_root=args.save_root)
    

