import pickle as pkl 
import json
import numpy as np
from pdb import set_trace as stop
import os
import math
from tqdm import tqdm
import mmap
import datetime
import glob
import pathlib
import collections
import multiprocessing


import argparse


# python generate_hhblits.py --parallel --dataset contact --steps 2 --function 2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--dataset', type=str,required=True)
parser.add_argument('--steps', type=str,required=True)
parser.add_argument('--function', type=str,required=True)
parser.add_argument('--threads', type=int,default=64)
args = parser.parse_args()

# datasets = ['contact','fluorescence','solubility','secondary']

if args.dataset == 'secondary':
    splits = ['train','valid','cb513']
elif args.dataset in ['biogrid','covid'] and args.function != '5':
    splits = ['unique_proteins']
else:
    splits = ['train','valid','test']
# splits = ['valid','test']

# datasets=['secondary']
# splits = ['train','valid','cb513']

tmp_file_name = 'tmpfiles/'+str(datetime.datetime.now()).replace(' ','_')

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def clean_string(input_string):
    input_string = input_string.replace('\n',' ')
    input_string = ''.join(input_string.split(' ')[2:])
    input_string = input_string.replace('\t',',')
    input_string = input_string.replace(',,',',')
    input_string = input_string.replace('*','inf')
    output_arr = input_string.split(',')[0:-1]
    return output_arr



def write_fasta_files(dataset,split):
    file_path = 'data/'+dataset+'/'+split+'.fasta'
    print(file_path)
    with open(file_path,'r') as f:
        seq_list = []

        k = 0
        total_lines=get_num_lines(file_path)
        for line in tqdm(f,total=total_lines):
            line = line.rstrip()
            if '>' in line:
                # line_id = line.split('>')[1]
                line_id = str(k)
            else:
                seq_list.append(line)
                fasta_file_name = 'hhblits_out/'+dataset+'/'+split+'/fasta/'+line_id+'.fasta'
                if not os.path.exists(fasta_file_name):
                    with open(fasta_file_name,'w') as f2:
                        f2.write('>'+line_id+'\n'+line)
                k+=1
    return seq_list

def generate_hmm_files(dataset,split,lock=False):

    for fasta_file_name in glob.glob('hhblits_out/'+dataset+'/'+split+'/fasta/*.fasta'):
        hmm_file_name = fasta_file_name.replace('fasta','hmm')

        if lock: lock.acquire()

        if not os.path.exists(hmm_file_name):
            pathlib.Path(hmm_file_name).touch()
            print(hmm_file_name)

            if lock: lock.release()

            cmd = 'hhblits -v 0 -i '+fasta_file_name+' -n '+args.steps+' -d /bigtemp/jjl5sw/hhblits/pfam/pfam -ohhm '+hmm_file_name
            # cmd = 'hhblits -v 0 -i '+fasta_file_name+' -n '+args.steps+' -d /bigtemp/jjl5sw/hhblits/UniRef30_2020_02 -ohhm '+hmm_file_name
            # cmd = 'hhblits -v 0 -i '+fasta_file_name+' -n '+args.steps+' -d /bigtemp/jjl5sw/hhblits/uniclust30_2016/uniclust30_2016_09 -ohhm '+hmm_file_name
            
            os.system(cmd)


            # Remove first N useless lines
            cmd = "sed -i '/HMM    /,$!d' "+hmm_file_name
            os.system(cmd)

            # cmd = "rm "+hmm_file_name.replace('.hmm','.hhr')
           # os.system(cmd)
        
        else:
            if lock: lock.release()


def convert_hmms_to_npy(dataset,split):
    output_dict = {}
    filelist = glob.glob('hhblits_out/'+dataset+'/'+split+'/hmm/*.hmm')
    for hmm_file_hame in filelist:
        # print(hmm_file_hame)
        line_num = hmm_file_hame.split('/')[-1].replace('.hmm','')
        # line_num = int(line_num)
        sample_list = []
        with open(hmm_file_hame,'r') as f3: 

            #burn first 3 lines
            f3.readline()
            f3.readline()
            f3.readline()

            while True:
                first_line = f3.readline()
                second_line = f3.readline()
                null_line = f3.readline()

                if first_line and second_line:
                    first_line_arr = clean_string(first_line)
                    second_line_arr = clean_string(second_line)
                    
                    full_line_arr = first_line_arr+second_line_arr
                    full_line_arr = [float(i) for i in full_line_arr] 
                    if len(full_line_arr)>0:
                        sample_list.append(full_line_arr)
                else:
                    break
        
        sample_arr = np.array(sample_list)

        
        try:
            assert sample_arr.shape[1] == 30
            sample_arr = np.power(2,-(sample_arr/1000))
        except:
            cmd = 'rm '+hmm_file_hame
            print('Error: ')
            print(cmd)
            os.system(cmd)

        
        output_dict[line_num] = sample_arr

    # stop()
    output_dict_sorted = collections.OrderedDict(sorted(output_dict.items()))
    
    output_list = [value for key,value in output_dict_sorted.items()]

    np.save('data/'+dataset+'/'+split+'.hmm',output_list)



def generate_hmm_files_wrapper(dataset,split,parallel=False):
    lock = multiprocessing.Lock()
    if parallel:
        jobs = []
        for i in range(args.threads):
            p = multiprocessing.Process(target=generate_hmm_files,args=(dataset,split,lock,))
            jobs.append(p)
            p.start()
        for job in jobs:
            p.join()
    else:
        generate_hmm_files(dataset,split)


def merge_hmm_and_json(dataset,split):
    save_flag = True
    json_file_name = 'data/'+dataset+'/'+split+'.json'
    print(json_file_name)
    new_list = []
    with open(json_file_name) as f:
        lines = json.load(f)
        hmm_file_name = json_file_name.replace('.json','.hmm.npy')
        hmm_data = np.load(hmm_file_name,allow_pickle=True)
        count=0
        for idx,line in enumerate(lines):
            try:
                line['hhblits'] = hmm_data[idx].tolist()
                assert len(line['hhblits']) == len(line['primary'])
            except:
                stop()
                save_flag = False
                print('ERROR: sample {}'.format(idx))
                cmd = 'rm hhblits_out/'+dataset+'/'+split+'/hmm/'+str(idx)+'.hmm'
                print(cmd)
                os.system(cmd)
            new_list.append(line)

    if save_flag:
        new_json_file_name = json_file_name = 'data/'+dataset+'/'+split+'_hhblits.json'
        print(new_json_file_name)
        with open(new_json_file_name, 'w') as fout:
            json.dump(new_list , fout)


def merge_hmm_and_paired_json(dataset,split):
    save_flag = True
    json_file_name = 'data/'+dataset+'/unique_proteins.json'
    print(json_file_name)
    new_list = []
    with open(json_file_name) as f:
        lines = json.load(f)
        hmm_file_name = json_file_name.replace('.json','.hmm.npy')
        hmm_data = np.load(hmm_file_name,allow_pickle=True)
        count=0
        for idx,line in enumerate(lines):
            try:
                for protein in ['protein_1','protein_2']:
                    protein_id = line['id']
                    line[protein]['hhblits'] = hmm_data[protein_id].tolist()
                assert len(line['hhblits']) == len(line['primary'])
            except:
                stop()
                save_flag = False
                print('ERROR: sample {}'.format(idx))
                cmd = 'rm hhblits_out/'+dataset+'/'+split+'/hmm/'+str(idx)+'.hmm'
                print(cmd)
                os.system(cmd)
            new_list.append(line)

    if save_flag:
        new_json_file_name = json_file_name = 'data/'+dataset+'/'+split+'_hhblits.json'
        print(new_json_file_name)
        with open(new_json_file_name, 'w') as fout:
            json.dump(new_list , fout)

if __name__ == "__main__":
    dataset = args.dataset
    if args.function == '1':
        #### WRITE FASTA FILE #######
        for split in splits:
            print('Gen Fasta: {} {}'.format(dataset,split))
            fasta_dir = os.path.join('hhblits_out',dataset,split,'fasta')
            if not os.path.exists(fasta_dir):
                os.system('mkdir -p '+fasta_dir)
            
            seq_list = write_fasta_files(dataset,split)

    elif args.function == '2':
        ###### GEN HMM FILES #######
        for split in splits:
            hmm_dir = os.path.join('hhblits_out',dataset,split,'hmm')
            if not os.path.exists(hmm_dir):
                os.system('mkdir -p '+hmm_dir)
            print('\nGen HMM: {} {}'.format(dataset,split))
            output_dir = os.path.join('hhblits_out',dataset,split)
            generate_hmm_files_wrapper(dataset,split,parallel=args.parallel)


    # Note: I had to make this execute separately because otherwise some of the
    # multithreaded processes finish running early and this exectutes too early
    elif args.function == '3':
        ##### GEN NPY FILES #######
        for split in splits:
            print('\nGen npy: {} {}'.format(dataset,split))
            output_dir = os.path.join('hhblits_out',dataset,split)

            # Note: for some reason the hmm generator creates empty hmm files
            # (might have something to do with paralellization). Need to remove
            # empty files using
            # find hhblits_out/stability/*/ -empty -type f -delete 
            # and then re-run generate_hmm_files_wrapper

            convert_hmms_to_npy(dataset,split)

    
    elif args.function == '4':
        ##### MERGE HMM and JSON #######
        for split in splits:
            print('\nGen npy: {} {}'.format(dataset,split))
            output_dir = os.path.join('hhblits_out',dataset,split)
            merge_hmm_and_json(dataset,split)
    

    elif args.function == '5':
        ##### MERGE HMM and PAIRED JSON FILES #######
        for split in splits:
            print('\nGen npy: {} {}'.format(dataset,split))
            output_dir = os.path.join('hhblits_out',dataset,split)
            merge_hmm_and_paired_json(dataset,split)
                    



# with open('data/biogrid_covid/all.json') as f:
#     lines = json.load(f)

# len_arr = []
# for sample in lines:
#     len_arr+=[len(sample['protein_1']['primary'])]
#     len_arr+=[len(sample['protein_2']['primary'])]

