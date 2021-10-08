import json
import numpy as np
import os
from glob import glob
from pdb import set_trace as stop
import random
import csv
from Bio import SeqIO
random.seed(17)

"""

"""

uniprot_fasta_name='data/uniprot/uniprot_sprot.fasta'
covid_fasta_name='data/covid/covid_proteins.fasta'
output_json_name='data/uniprot/sprot_and_covid.json'
SARSCOV2_ID='2697049'
HUMAN_ID='9606'



def create_seq_dict(fasta_file):
    uniprot_dict = {}
    with open(fasta_file,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            prot_id = line[0]
            prot_seq = line[1]
            uniprot_dict[prot_id] = prot_seq

    return uniprot_dict



if __name__ == "__main__":
    output_list = []
    seq_dict = create_seq_dict('/af11/jjl5sw/HVPPI/Protein_seq.tsv')
    pair_seq_file = '/af11/jjl5sw/HVPPI/Protein_pair.tsv'
    output_json_name = '/af11/jjl5sw/HVPPI/test.json'

    
    with open(pair_seq_file,'r') as f:
        for line in f:
            line = line.strip().split('\t')

            human_protein = line[0]
            virus_protein = line[1]

            human_protein_seq = seq_dict[human_protein]
            virus_protein_seq = seq_dict[virus_protein]
            is_interaction = int(line[2])

            sample = {}
            sample['protein_1'] = {'id':virus_protein,'primary':virus_protein_seq}
            sample['protein_2'] = {'id':human_protein,'primary':human_protein_seq}
            sample['is_interaction'] = is_interaction
            output_list.append(sample)

    
    with open(output_json_name, 'w') as fout:
        json.dump(output_list , fout)
    
