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
output_json_name='data/uniprot/uniprot_sprot.json'

# uniprot_fasta_name='data/uniprot/uniref50.fasta'
# output_json_name='data/uniprot/uniref50.json'


# covid_fasta_name='data/covid/covid_proteins.fasta'
# output_json_name='data/uniprot/sprot_and_covid.json'
SARSCOV2_ID='2697049'
HUMAN_ID='9606'

# uniprot_fasta_name='data/sarscov2/data/cov/cov_all.fa'
# output_json_name='data/sarscov2/data/cov/cov_all.json'



def create_uniprot_dict(fasta_file):
    uniprot_dict = {}
    for seq_record in SeqIO.parse(uniprot_fasta_name, "fasta"):
        uniprot_id = str(seq_record.id)
        if '|' in uniprot_id:
            uniprot_id = uniprot_id.split('|')[1]
        sequence = str(seq_record.seq)
        # species = str(seq_record.description).split('OX=')[1].split(' ')[0]
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


if __name__ == "__main__":
    output_list = []
    uniprot_dict = create_uniprot_dict(uniprot_fasta_name)
    # covid_dict = create_covid_dict(covid_fasta_name)

    for key,value in uniprot_dict.items():
        sample = {}
        sample['id'] = key
        sample['primary'] = value
        sample['family'] = ''
        output_list.append(sample)
    
    # stop()

    with open(output_json_name, 'w') as fout:
        json.dump(output_list , fout)
    
