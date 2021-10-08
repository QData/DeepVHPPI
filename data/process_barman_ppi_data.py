
import pickle as pkl 
import json
import numpy as np
import os
import csv
import pandas as pd
from pdb import set_trace as stop
from glob import glob
import random
from Bio import SeqIO
import copy
import argparse
import random

file_root = 'data/barman_ppi/'



def create_protein_to_seq_dict(ppi_file):
    protein_to_seq = {}
    for line in open(ppi_file,'r').readlines()[1:]:
        line = line.strip().split(',')
        HOST_TAXID = line[0]
        VIRUS_TAXID = line[1]
        HOST = line[2]
        VIRUS = line[3]
        HOST_SEQ = line[4]
        VIRUS_SEQ = line[5]
        if HOST not in protein_to_seq:
            protein_to_seq[HOST] = HOST_SEQ
        if VIRUS not in protein_to_seq:
            protein_to_seq[HOST] = VIRUS_SEQ

    return protein_to_seq

def add_to_protein_dict(protein_to_seq,fasta_file,species=None):
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        uniprot_id = str(seq_record.id).split('|')[1]
        sequence = str(seq_record.seq)
        protein_to_seq[uniprot_id] = sequence
    return protein_to_seq

def create_split_list(split_list,file_name,uniprot_dict,is_interaction):
    missing_human = {}
    missing_virus = {}
    
    for line in open(file_name,'r').readlines()[1:]:
        line = line.strip().split(',')
        virus_protein = line[0]
        human_protein = line[1]
        human_protein_seq = False
        virus_protein_seq = False

        if human_protein in uniprot_dict:
            human_protein_seq = uniprot_dict[human_protein]
        else:
            if human_protein not in missing_human:
                missing_human[human_protein] = True
                print('H:',human_protein)
        
        if virus_protein in uniprot_dict:
            virus_protein_seq = uniprot_dict[virus_protein]
        else:
            if virus_protein not in missing_virus:
                missing_virus[virus_protein] = True
                print('V:',virus_protein)
        
        if human_protein_seq and virus_protein_seq:
            sample = {}
            sample['protein_1'] = {'id':virus_protein,'primary':virus_protein_seq}
            sample['protein_2'] = {'id':human_protein,'primary':human_protein_seq}
            sample['is_interaction'] = is_interaction
            split_list.append(sample)
    
    # print('H Total:',len(missing_human))
    # print('V Total:',len(missing_virus))
    # stop()
    return split_list




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--save_root", type=str, default='data/coronavirus/', help="")
    args = parser.parse_args()


    # all_ppis_file = os.path.join(file_root,'barman_ppis.csv')
    # protein_to_seq = create_protein_to_seq_dict(all_ppis_file)

    protein_to_seq = {}
    uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
    barman_fasta_name='data/uniprot/barman_extra.fasta'
    protein_to_seq = add_to_protein_dict(protein_to_seq,uniprot_fasta_name)
    protein_to_seq = add_to_protein_dict(protein_to_seq,barman_fasta_name)
    

    pos_file =  os.path.join(file_root,'barman_pos.csv')
    neg_file =  os.path.join(file_root,'barman_neg.csv')


    full_list = []
    full_list = create_split_list(full_list,pos_file,protein_to_seq,1)
    full_list = create_split_list(full_list,neg_file,protein_to_seq,0)

    random.shuffle(full_list)

    split_lens = int(len(full_list)/5)

    splits = []
    splits.append(full_list[0:split_lens*1])
    splits.append(full_list[split_lens*1:split_lens*2])
    splits.append(full_list[split_lens*2:split_lens*3])
    splits.append(full_list[split_lens*3:split_lens*4])
    splits.append(full_list[split_lens*4:])

    train1 = splits[0]+splits[1]+splits[2]+splits[3]
    test1 = splits[4]

    train2 = splits[1]+splits[2]+splits[3]+splits[4]
    test2 = splits[0]

    train3 = splits[0]+splits[2]+splits[3]+splits[4]
    test3 = splits[1]

    train4 = splits[0]+splits[1]+splits[3]+splits[4]
    test4 = splits[2]

    train5 = splits[0]+splits[1]+splits[2]+splits[4]
    test5 = splits[3]


    print(file_root+'/train1.json')
    with open(file_root+'/train1.json', 'w') as fout:
        json.dump(train1 , fout)
    print(file_root+'/test1.json')
    with open(file_root+'/test1.json', 'w') as fout:
        json.dump(test1 , fout)

    print(file_root+'/train2.json')
    with open(file_root+'/train2.json', 'w') as fout:
        json.dump(train2 , fout)
    print(file_root+'/test2.json')
    with open(file_root+'/test2.json', 'w') as fout:
        json.dump(test2 , fout)

    print(file_root+'/train3.json')
    with open(file_root+'/train3.json', 'w') as fout:
        json.dump(train3 , fout)
    print(file_root+'/test3.json')
    with open(file_root+'/test3.json', 'w') as fout:
        json.dump(test3 , fout)

    print(file_root+'/train4.json')
    with open(file_root+'/train4.json', 'w') as fout:
        json.dump(train4 , fout)
    print(file_root+'/test4.json')
    with open(file_root+'/test4.json', 'w') as fout:
        json.dump(test4 , fout)

    print(file_root+'/train5.json')
    with open(file_root+'/train5.json', 'w') as fout:
        json.dump(train5 , fout)
    print(file_root+'/test5.json')
    with open(file_root+'/test5.json', 'w') as fout:
        json.dump(test5 , fout)
