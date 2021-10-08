
import pickle as pkl 
import json
import numpy as np
import os
import csv
import pandas as pd
from Bio import SeqIO
from pdb import set_trace as stop
import copy

# file_root = 'data/zhou_ppi/h1n1/host/'
file_root = 'data/yang_ppi/'

def create_mapping_dict(file_name):
    mapping_dict = {}

    for line in open(file_name,'r').readlines()[1:]:
        line = line.strip().split('\t')
        new_id = line[0]
        for old_id in line[-1].split(','):
            mapping_dict[old_id] = new_id

    return mapping_dict

def get_unique_proteins(file_list,output_name):
    unique_proteins = {}
    for file in file_list:
        for line in open(file,'r').readlines()[1:]:
            line = line.strip().split('\t')
            host_protein = line[1].split('-')[0]
            virus_protein = line[2].split('-')[0]

            if host_protein not in unique_proteins:
                unique_proteins[host_protein] = True
            if virus_protein not in unique_proteins:
                unique_proteins[virus_protein] = True

    with open(output_name,'w') as f:
        for protein in unique_proteins:
            f.write(protein+'\n')


def parse_nih(entry):
    fields = entry.split('|')

    country = fields[3]
    # from locations import country2continent
    # if country in country2continent:
    #     continent = country2continent[country]
    # else:
    #     country = 'NA'
    #     continent = 'NA'
    country = 'NA'
    continent = 'NA'

    meta = {
        'strain': 'SARS-CoV-2',
        'host': 'human',
        'group': 'human',
        'country': country,
        'continent': continent,
        'dataset': 'nih',
    }
    return meta

def get_spike_seqs(fasta_file):
    seqs = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        if len(record.seq) < 1000:
            continue
        if str(record.seq).count('X') > 0:
            continue
        if record.seq not in seqs:
            seqs[record.seq] = []
        if fasta_file == 'data/cov/viprbrc_db.fasta':
            meta = parse_viprbrc(record.description)
        elif fasta_file == 'data/cov/gisaid.fasta':
            meta = parse_gisaid(record.description)
        else:
            meta = parse_nih(record.description)
        meta['accession'] = record.description
        seqs[record.seq].append(meta)

    seqs = list(seqs.keys())
    seqs = [str(seq) for seq in seqs]
    return seqs

def create_protein_to_seq(fasta_file,species=None):
    protein_to_seq = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):

        uniprot_id = str(seq_record.id).split('|')[1]
        sequence = str(seq_record.seq)
        protein_to_seq[uniprot_id] = sequence
    return protein_to_seq

def create_covid_protein_to_seq(fasta_file,species=None):
    protein_to_seq = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        uniprot_id = str(seq_record.id).split('|')[1]
        description = str(seq_record.description)
        if ('OX=9606' in description) or ('OX=694009' in description):
            sequence = str(seq_record.seq)
            protein_to_seq[uniprot_id] = sequence

    return protein_to_seq



def create_split_list(file_name,protein_to_seq,mapping_dict):
    split_list = []
    missing_count = 0
    for line in open(file_name,'r').readlines()[1:]:
        line = line.strip().split('\t')

        if line[0] == "+1":
            is_interaction = 1
        else:
            is_interaction = 0
        
        host_protein = line[1].split('-')[0]
        virus_protein = line[2].split('-')[0]

        

        new_host_id = mapping_dict[host_protein]
        new_virus_id = mapping_dict[virus_protein]

        virus_protein_seq = False
        host_protein_seq = False

        if new_host_id in protein_to_seq:
            host_protein_seq = protein_to_seq[new_host_id]
        else:
            pass
            # print('H:',host_protein)
        
        if new_virus_id in protein_to_seq:
            virus_protein_seq = protein_to_seq[new_virus_id]
        else:
            pass
            # print('V:',virus_protein)
        
        if virus_protein_seq and host_protein_seq:
            sample = {}
            sample['protein_1'] = {'id':virus_protein,'primary':virus_protein_seq}
            sample['protein_2'] = {'id':host_protein,'primary':host_protein_seq}
            sample['is_interaction'] = is_interaction
            split_list.append(sample)
        else:
            missing_count +=1

    print('Missing: ',missing_count)
    return split_list

train_file =  os.path.join(file_root,'train_set_group1')
test_file =  os.path.join(file_root,'independent_test_group1')

get_unique_proteins([train_file,test_file],file_root+'/unique_proteins.txt')


# uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
fasta_name='data/yang_ppi/unique_proteins.fasta'
mapping_file_name='data/yang_ppi/mapping_list.tab'

protein_to_seq = create_protein_to_seq(fasta_name)


mapping_dict = create_mapping_dict(mapping_file_name)

train_list = create_split_list(train_file,protein_to_seq,mapping_dict)
test_list = create_split_list(test_file,protein_to_seq,mapping_dict)


train_test_list = train_list+test_list
print(len(train_test_list))


print(file_root+'/train.json')
with open(file_root+'/train.json', 'w') as fout:
    json.dump(train_list , fout)

print(file_root+'/test.json')
with open(file_root+'/test.json', 'w') as fout:
    json.dump(test_list , fout)

print(file_root+'/train_test.json')
with open(file_root+'/train_test.json', 'w') as fout:
    json.dump(train_test_list , fout)




spike_seqs = get_spike_seqs('data/SARSCoV2_mutations/data/cov/cov_all.fa')
covid_protein_to_seq = create_covid_protein_to_seq(fasta_name)
mapping_dict = create_mapping_dict(mapping_file_name)

train_list = create_split_list(train_file,covid_protein_to_seq,mapping_dict)
test_list = create_split_list(test_file,covid_protein_to_seq,mapping_dict)


train_test_list = train_list+test_list
print(len(train_test_list))

new_samples = []
k=0
for sample in train_test_list: 
    if sample['protein_1']['id'] == 'P59594':
        new_samples.append(sample)
        if sample['protein_2']['id'] == 'Q9BYF1':
            for i,seq in enumerate(spike_seqs):
                new_sample = copy.deepcopy(sample)
                new_sample['protein_1']['id'] = 'spike'+str(i+1)
                new_sample['protein_1']['primary'] = seq
                new_samples.append(new_sample)
        else:
            k+=1



print(file_root+'/covid_train_test.json')
with open(file_root+'/train_test.json', 'w') as fout:
    json.dump(new_samples , fout)