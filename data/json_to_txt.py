
import lmdb 
import pickle as pkl 
import json
import numpy as np
import os
from glob import glob
from pdb import set_trace as stop


for directory in glob("./pfam/"):
    print(directory)
    for file_name in glob(os.path.join(directory,"*.json")):
        # if 'train' not in file_name:
        try:
            with open(file_name) as f:
                lines = json.load(f)
                print(file_name.replace('.json','.raw'))
                fasta_file_name = file_name.replace('.json','.raw')
                fasta_file = open(fasta_file_name,'w')
                count=0
                for line in lines:
                    primary = line['primary']
                    for aa in primary[0:-1]:
                        fasta_file.write(str(aa)+' ')
                    fasta_file.write(str(primary[-1]))
                    fasta_file.write('\n\n')
                    count+=1
                fasta_file.close()
        except UnicodeDecodeError:
            # stop()
            pass