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
SARSCOV2_ORG_ID='2697049'
HUMAN_ORG_ID='9606'


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

def get_sp_id(swiss_prot_ids,uniprot_dict):
    for sp_id in swiss_prot_ids:
        if sp_id in uniprot_dict:
            return sp_id,uniprot_dict[sp_id]
    return False,False

def create_v_h_dict(virus_protein_dict,human_protein_dict):
    """
    All possible v-h interactions len(virus_protein_dict)*len(human_protein_dict) total
    v_h_dict[virus_protein] = {human_protein_sp_id1,human_protein_sp_id2,...}
    """
    v_h_dict = {}
    for virus_protein in virus_protein_dict:
        v_h_dict[virus_protein] = {}
        for human_protein_sp_id in human_protein_dict:
            v_h_dict[virus_protein][human_protein_sp_id] = 0
    return v_h_dict

def add_positives_to_v_h_dict(covid_biogrid_file,v_h_dict,uniprot_dict,virus_protein_dict,human_protein_dict):
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

            protein_a_sp_id,protein_a_seq = get_sp_id(protein_a_sp_list,uniprot_dict)
            protein_b_sp_id,protein_b_seq = get_sp_id(protein_b_sp_list,uniprot_dict)

            if (protein_a_organism == SARSCOV2_ORG_ID and protein_b_organism == HUMAN_ORG_ID) or (protein_a_organism == HUMAN_ORG_ID and protein_b_organism == SARSCOV2_ORG_ID):
                
                if protein_a_organism == SARSCOV2_ORG_ID:
                    if protein_a_symbol in virus_protein_dict and protein_b_sp_id in human_protein_dict:
                        v_h_dict[protein_a_symbol][protein_b_sp_id] = 1
                elif protein_b_organism == SARSCOV2_ORG_ID:
                    if protein_b_symbol in virus_protein_dict and protein_a_sp_id in human_protein_dict:
                        v_h_dict[protein_b_symbol][protein_a_sp_id] = 1

                # if protein_a_organism == SARSCOV2_ORG_ID and protein_a_symbol not in virus_protein_dict:
                #     print('V',protein_a_symbol)
                # if protein_b_organism == SARSCOV2_ORG_ID and protein_b_symbol not in virus_protein_dict:
                #     print('V',protein_b_symbol)
                # if protein_a_organism != SARSCOV2_ORG_ID and protein_a_symbol not in human_protein_dict:
                #     print('H',protein_a_symbol)
                # if protein_b_organism != SARSCOV2_ORG_ID and protein_b_symbol not in human_protein_dict:
                #     print('H',protein_b_symbol)
                    

    print('# Not Found: ',non_found_count)
    return v_h_dict


def create_interaction_list(v_h_dict,virus_protein_dict,unique_human_proteins):
    interaction_list = []
    pos_count = 0
    neg_count = 0
    for virus_protein in v_h_dict:
        for human_protein in v_h_dict[virus_protein]:
            is_interaction = v_h_dict[virus_protein][human_protein]
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

    pair_file = open(os.path.join(save_root,'Protein_pair.tsv'),'w')
    sars_cov2_fasta = open(os.path.join(save_root,'sars_cov2.fa'),'w')
    human_fasta = open(os.path.join(save_root,'human.fa'),'w')
    seq_file = open(os.path.join(save_root,'Protein_seq.tsv'),'w')

    sars_cov2_ids = {}
    human_ids = {}
    all_ids = {}
    for sample in interaction_list:
        is_interaction = sample['is_interaction']
        p1_id = sample['protein_1']['id']
        p1_seq = sample['protein_1']['primary']
        p2_id = sample['protein_2']['id']
        p2_seq = sample['protein_2']['primary']

        # if p1_id == 'S':

        if len(p1_seq)>30 and len(p2_seq)>30:
            pair_file.write(p2_id+'\t'+p1_id+'\t'+str(is_interaction)+'\n')

            if p1_id not in all_ids:
                all_ids[p1_id] = p1_seq
                if len(p1_seq)>4999:
                    p1_seq = p1_seq[0:4999]
                seq_file.write(p1_id+'\t'+p1_seq+'\n')
            if p2_id not in all_ids:
                all_ids[p2_id] = p2_seq
                if len(p2_seq)>4999:
                    p2_seq = p2_seq[0:4999]
                seq_file.write(p2_id+'\t'+p2_seq+'\n')
            
            if p1_id not in sars_cov2_ids:
                sars_cov2_ids[p1_id] = p1_seq
                if len(p1_seq)>4999:
                    p1_seq = p1_seq[0:4999]
                sars_cov2_fasta.write('>'+p1_id+'\n'+p1_seq+'\n')
            if p2_id not in human_ids:
                human_ids[p2_id] = p2_seq
                if len(p2_seq)>4999:
                    p2_seq = p2_seq[0:4999]
                human_fasta.write('>'+p2_id+'\n'+p2_seq+'\n')

    

    sars_cov2_fasta.close()
    human_fasta.close()
    pair_file.close()
    seq_file.close()


    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default='/af11/jjl5sw/HVPPI/', help="")
    args = parser.parse_args()

    print('\n=====> Create uniprot dict')
    uniprot_dict = create_uniprot_dict(uniprot_fasta_name)

    print('\n=====> Get covid protein dict')
    virus_protein_dict = create_covid_dict(covid_fasta_name)

    print('\n=====> Get human protein dict')
    human_protein_dict = create_uniprot_dict(uniprot_fasta_name,HUMAN_ORG_ID)

    print('# Human Proteins: ',len(human_protein_dict))
    print('# Covid Proteins: ',len(virus_protein_dict))
    
    print('\n=====> Create V-H interaction graph')
    v_h_dict = create_v_h_dict(virus_protein_dict,human_protein_dict)

    print('\n=====> Add positive pairs to H-V interaction graph')
    v_h_dict = add_positives_to_v_h_dict(covid_biogrid_file,v_h_dict,uniprot_dict,virus_protein_dict,human_protein_dict)

    print('\n=====> Create interaction list')
    interaction_list = create_interaction_list(v_h_dict,virus_protein_dict,human_protein_dict)
    

    ordered_v_h_dict = OrderedDict(sorted(v_h_dict.items()))
    for virus_protein in ordered_v_h_dict.keys():
        pos_count = 0
        for human_protein in ordered_v_h_dict[virus_protein]:
            if ordered_v_h_dict[virus_protein][human_protein] ==1:
                pos_count+=1
        print(virus_protein,',',pos_count)



    print('\n=====> Save splits')
    create_and_save_spilts(interaction_list,args.save_root)

    