
import pickle as pkl 
import json
import numpy as np
import os
import csv
import pandas as pd
from pdb import set_trace as stop

# file_root = 'data/zhou_ppi/h1n1/human/'
file_root = 'data/zhou_ppi/ebola/human/'

def create_split_list(split_list,file_name,is_interaction):
    for line in open(file_name,'r').readlines()[1:]:
        line = line.strip().split(',')
        human_protein = line[0]
        virus_protein = line[1]
        human_protein_seq = line[2]
        virus_protein_seq = line[3]
        sample = {}
        sample['protein_1'] = {'id':virus_protein,'primary':virus_protein_seq}
        sample['protein_2'] = {'id':human_protein,'primary':human_protein_seq}
        sample['is_interaction'] = is_interaction
        split_list.append(sample)
    return split_list


train_pos =  os.path.join(file_root,'train_pos.csv')
train_neg =  os.path.join(file_root,'train_neg.csv')

test_pos =  os.path.join(file_root,'test_pos.csv')
test_neg =  os.path.join(file_root,'test_neg.csv')

# train_pos_df = pd.read_csv(train_pos, sep=',')
# train_neg_df = pd.read_csv(train_neg, sep=',')
# test_pos_df = pd.read_csv(test_pos, sep=',')
# test_neg_df = pd.read_csv(test_pos, sep=',')


train_list = []
train_list = create_split_list(train_list,train_pos,1)
train_list = create_split_list(train_list,train_neg,0)

test_list = []
test_list = create_split_list(test_list,test_pos,1)
test_list = create_split_list(test_list,test_neg,0)


print(file_root+'/train.json')
with open(file_root+'/train.json', 'w') as fout:
    json.dump(train_list , fout)

print(file_root+'/test.json')
with open(file_root+'/test.json', 'w') as fout:
    json.dump(test_list , fout)