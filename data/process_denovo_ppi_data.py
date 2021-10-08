
import pickle as pkl 
import json
import numpy as np
import os
import csv
import pandas as pd
from pdb import set_trace as stop
from Bio import SeqIO

# file_root = 'data/zhou_ppi/h1n1/human/'
file_root = 'data/DeNovo/'

uniprot_fasta_name = 'data/DeNovo/all_proteins.fasta'

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

def create_split_list(protein_dict,split_list,file_name,is_interaction):
    for line in open(file_name,'r').readlines()[1:]:
        line = line.strip().split(',')
        human_protein = line[0]
        virus_protein = line[1]

        human_protein_seq = protein_dict[human_protein]
        virus_protein_seq = protein_dict[virus_protein]
        sample = {}
        sample['protein_1'] = {'id':virus_protein,'primary':virus_protein_seq}
        sample['protein_2'] = {'id':human_protein,'primary':human_protein_seq}
        sample['is_interaction'] = is_interaction
        split_list.append(sample)
    return split_list


protein_dict = create_uniprot_dict(uniprot_fasta_name)

train_pos =  os.path.join(file_root,'train_pos.csv')
train_neg =  os.path.join(file_root,'train_neg.csv')

test_pos =  os.path.join(file_root,'test_pos.csv')
test_neg =  os.path.join(file_root,'test_neg.csv')

# train_pos_df = pd.read_csv(train_pos, sep=',')
# train_neg_df = pd.read_csv(train_neg, sep=',')
# test_pos_df = pd.read_csv(test_pos, sep=',')
# test_neg_df = pd.read_csv(test_pos, sep=',')


train_list = []
train_list = create_split_list(protein_dict,train_list,train_pos,1)
train_list = create_split_list(protein_dict,train_list,train_neg,0)

test_list = []
test_list = create_split_list(protein_dict,test_list,test_pos,1)
test_list = create_split_list(protein_dict,test_list,test_neg,0)


print(file_root+'/train.json')
with open(file_root+'/train.json', 'w') as fout:
    json.dump(train_list , fout)

print(file_root+'/test.json')
with open(file_root+'/test.json', 'w') as fout:
    json.dump(test_list , fout)