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

RUN_NUM='1'
random.seed(17) # run 1

# RUN_NUM='2'
# random.seed(27) # run 2

# RUN_NUM='3'
# random.seed(37) # run 3

"""
For Generating covid ppi data from Covid BioGRID 

10/26/20 update: this file is not used for final results
"""

 
human_biogrid_file = 'data/human_ppi/BIOGRID-ORGANISM-Homo_sapiens-4.1.190.tab3.txt'
covid_biogrid_file = 'data/coronavirus_ppi/BIOGRID-CORONAVIRUS-4.1.190.tab3.txt'

uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
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

def get_unique_species_proteins(input_biogrid_file,uniprot_dict,species_id):
    """
    Returns list of human proteins from input file
    """
    unique_species_proteins = {}
    with open(input_biogrid_file, 'r') as f:
        d_reader = csv.DictReader(f,delimiter='\t')
        headers = d_reader.fieldnames

        for line in d_reader:
            protein_a_organism = line['Organism ID Interactor A']
            protein_b_organism = line['Organism ID Interactor B']
            protein_a_sp_list = line['SWISS-PROT Accessions Interactor A'].split('|')
            protein_b_sp_list = line['SWISS-PROT Accessions Interactor B'].split('|')

            if protein_a_organism == species_id:
                protein_a_sp_id,protein_a_seq = get_sp_id(protein_a_sp_list,uniprot_dict)
                protein_b_sp_id,protein_b_seq = get_sp_id(protein_b_sp_list,uniprot_dict)
                if protein_a_seq and protein_b_seq and (protein_a_sp_id not in unique_species_proteins):
                    unique_species_proteins[protein_a_sp_id] = protein_a_seq

            if protein_b_organism == species_id:
                protein_b_sp_id,protein_b_seq = get_sp_id(protein_b_sp_list,uniprot_dict)
                protein_a_sp_id,protein_a_seq = get_sp_id(protein_a_sp_list,uniprot_dict)
                if protein_b_seq and protein_a_seq and (protein_b_sp_id not in unique_species_proteins):
                    unique_species_proteins[protein_b_sp_id] = protein_b_seq

    # unique_species_proteins = list(unique_species_proteins.keys())
    return unique_species_proteins

def get_unique_proteins(covid_biogrid_file,uniprot_dict,covid_dict):
    virus_protein_dict = {}
    unique_human_proteins = {}
    idx = 0
    HH_count = 0
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

            if protein_a_organism == protein_b_organism == HUMAN_ORG_ID:
                HH_count += 1

            if protein_a_organism != protein_b_organism: #only want H-V interactions

                if protein_a_organism == SARSCOV2_ORG_ID:
                    protein_a_seq = covid_dict[protein_a_symbol]
                    if protein_a_symbol not in virus_protein_dict:
                        virus_protein_dict[protein_a_symbol] = protein_a_seq
                elif protein_a_organism == HUMAN_ORG_ID:
                    protein_a_sp_id,protein_a_seq = get_sp_id(protein_a_sp_list,uniprot_dict)
                    if protein_a_seq and (protein_a_sp_id not in unique_human_proteins):
                        unique_human_proteins[protein_a_sp_id] = protein_a_seq

                if protein_b_organism == SARSCOV2_ORG_ID: 
                    protein_b_seq = covid_dict[protein_b_symbol]
                    if protein_b_symbol not in virus_protein_dict:
                        virus_protein_dict[protein_b_symbol] = protein_b_seq
                elif protein_b_organism == HUMAN_ORG_ID:
                    protein_b_sp_id,protein_b_seq = get_sp_id(protein_b_sp_list,uniprot_dict)
                    if protein_b_seq and (protein_b_sp_id not in unique_human_proteins):
                        unique_human_proteins[protein_b_sp_id] = protein_b_seq
            

    return virus_protein_dict,unique_human_proteins


def create_v_h_dict(virus_protein_dict,human_protein_dict):
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

                flag = False
                
                if protein_a_organism == SARSCOV2_ORG_ID:
                    if protein_a_sp_id in virus_protein_dict:
                        if protein_b_sp_id in v_h_dict[protein_a_sp_id]:
                            v_h_dict[protein_a_sp_id][protein_b_sp_id] = 1
                            flag = True
                elif protein_b_organism == SARSCOV2_ORG_ID:
                    if protein_b_sp_id in v_h_dict:
                        if protein_a_sp_id in v_h_dict[protein_b_sp_id]:
                            v_h_dict[protein_b_sp_id][protein_a_sp_id] = 1
                            flag = True
                else:
                    flag = True

                
                if flag is False:
                    non_found_count +=1
                    # print(protein_a_organism,protein_b_organism)
                    # print(protein_a_symbol,protein_b_symbol)
                    # print(protein_a_sp_list,protein_b_sp_list)
                    # print(protein_a_sp_id,protein_b_sp_id)
                    # print('-------')
                    # stop()
    print('# Not Found: ',non_found_count)
    return v_h_dict


def create_interaction_list(v_h_dict,virus_protein_dict,human_protein_dict):
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
            human_protein_seq = human_protein_dict[human_protein]
            
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
    # parser.add_argument("--save_root", type=str, default='data/coronavirus_ppi/', help="")
    parser.add_argument("--save_root", type=str, default='/af11/jjl5sw/HVPPI/', help="")
    args = parser.parse_args()

    print('\n=====> Create uniprot dict')
    uniprot_dict = create_uniprot_dict(uniprot_fasta_name)

    print('\n=====> Get covid protein dict')
    virus_protein_dict = create_uniprot_dict(uniprot_fasta_name,SARSCOV2_ORG_ID)

    print('\n=====> Get human protein dict')
    # human_protein_dict = get_unique_species_proteins(covid_biogrid_file,uniprot_dict,HUMAN_ORG_ID)
    human_protein_dict = create_uniprot_dict(uniprot_fasta_name,HUMAN_ORG_ID)

    print('# Human Proteins: ',len(human_protein_dict))
    print('# Covid Proteins: ',len(virus_protein_dict))
    
    print('\n=====> Create H-V interaction graph')
    v_h_dict = create_v_h_dict(virus_protein_dict,human_protein_dict)
    # stop()

    print('\n=====> Add positive pairs to H-V interaction graph')
    v_h_dict = add_positives_to_v_h_dict(covid_biogrid_file,v_h_dict,uniprot_dict,virus_protein_dict,human_protein_dict)

    print('\n=====> Create interaction list')
    interaction_list = create_interaction_list(v_h_dict,virus_protein_dict,human_protein_dict)

    print('\n=====> Save splits')
    create_and_save_spilts(interaction_list,args.save_root)

    
