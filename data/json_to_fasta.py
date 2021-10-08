
import lmdb 
import pickle as pkl 
import json
import numpy as np
import os
from glob import glob
from pdb import set_trace as stop

root_dir = 'data/biogrid/'


def write_single_json():
    for file_name in glob(root_dir+"/*.json"):
        print(file_name)
        with open(file_name) as f:
            lines = json.load(f)
            print(file_name.replace('.json','.fasta'))
            fasta_file_name = file_name.replace('.json','.fasta')
            fasta_file = open(fasta_file_name,'w')
            count=0
            for line in lines:
                primary = line['primary']
                if 'id' in line:
                    fasta_file.write('>'+str(line['id'])+'\n')
                else:
                    fasta_file.write('>seq'+str(count)+'\n')
                for aa in primary:
                    fasta_file.write(str(aa))
                fasta_file.write('\n')
                count+=1
            fasta_file.close()

def write_paired_json():
    fasta_file_name = root_dir+'/unique_proteins.fasta'
    fasta_file = open(fasta_file_name,'w')
    ids = {}
    for file_name in glob(root_dir+"/*.json"):
        print(file_name)
        with open(file_name) as f:
            lines = json.load(f)
            count=0
            for line in lines:
                for protein in ['protein_1','protein_2']:
                    primary = line[protein]['primary']
                    prot_id = line[protein]['id']
                    if prot_id not in ids:
                        ids[prot_id] = ''
                        fasta_file.write('>'+str(prot_id)+'\n')
                        for aa in primary:
                            fasta_file.write(str(aa))
                        fasta_file.write('\n')
                        count+=1
    fasta_file.close()


write_paired_json()
# write_single_json()
    


    
