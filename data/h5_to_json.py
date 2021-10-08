import json
import numpy as np
import os
import glob
from pdb import set_trace as stop
import random
import csv
from Bio import SeqIO
import pandas as pd

random.seed(17)

"""

"""

def list_to_json(output_json_name,output_list):
    print(output_json_name)   
    with open(output_json_name, 'w') as fout:
        json.dump(output_list , fout)

if __name__ == "__main__":
    
    for file_name in glob.glob('data/malaria/*.h5'):
        output_list = []
        
        print(file_name)
        df = pd.read_hdf(file_name, "table")
        print(df.keys())

        for idx in df.index:            
            sample = {}
            sample['primary'] = df['sequence'][idx]
            label = df['binding_site'][idx]
            label = [int(x) for x in label]
            sample['label'] = label
            sample['domain'] = df['cath_domain'][idx]
            sample['id'] = df['cathID'][idx]
            output_list.append(sample)
    

        output_json_name = file_name.replace('.h5','.json')
        output_json_name = output_json_name.replace('full-','')
        output_json_name = output_json_name.replace('-0.8','')
        output_json_name = output_json_name.replace('-0.2','')
        output_json_name = output_json_name.replace('-split','')
        output_json_name = output_json_name.replace('validation','valid')
        output_json_name = output_json_name.replace('-','_')

        if 'validation' in file_name:
            # half_len = int(len(output_list)/2)
            # valid_list = output_list[0:half_len]
            # test_list = output_list[half_len:]

            # list_to_json(output_json_name,valid_list)
            # list_to_json(output_json_name.replace('valid','test'),test_list)
            
            list_to_json(output_json_name,output_list)

            list_to_json(output_json_name.replace('valid','test'),output_list)
            
        else:
            list_to_json(output_json_name,output_list)
        
        # if file_name == 'data/malaria/full-train-split-S100-0.8.h5':
        #     stop()
            
    
