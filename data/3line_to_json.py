
import lmdb 
import pickle as pkl 
import json
import numpy as np

dataset = 'transmembrane'

for split in ['train','valid','test']:

    intput_file_name = 'data/'+dataset+'/'+dataset+'_'+split+'.3line'
    output_fasta_file_name = 'data/'+dataset+'/'+split+'.fasta'
    output_fasta_file = open(output_fasta_file_name,'w')

    output_list = []
    sample = None

    label_dict = {}
    label_dict_idx_count = 0

    with open(intput_file_name) as f:
        while True:
            first_line = f.readline()
            second_line = f.readline()
            third_line = f.readline()

            if first_line and second_line and third_line:
                first_line = first_line.strip()
                second_line = second_line.strip()
                third_line = third_line.strip()

                sample = {}
                sample['id'] = first_line.split('>')[1]
                sample['primary'] = second_line

                sample_output = []
                for char in third_line:
                    if char not in label_dict:
                        label_dict[char] = label_dict_idx_count
                        label_dict_idx_count+=1

                    sample_output.append(label_dict[char])

                sample['transmembrane'] = sample_output

                output_list.append(sample)
                output_fasta_file.write(first_line+'\n')
                output_fasta_file.write(second_line+'\n')
            else:
                break
    
    output_fasta_file.close()


    
    with open('data/'+dataset+'/'+split+'.json', 'w') as fout:
        json.dump(output_list , fout)

print(len(label_dict))
