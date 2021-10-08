import json
from pdb import set_trace as stop

data = json.load(open('data/yang_ppi/train_test.json','r'))
# data = json.load(open('data/yang_ppi/train.json','r'))


mapping_file_name='data/yang_ppi/mapping_list.tab'


def create_mapping_dict(file_name):
    '''Takes in a file name, maps to dictionary with information'''
    mapping_dict = {}

    for line in open(file_name,'r').readlines()[1:]:
        line = line.strip().split('\t')
        new_id = line[0]
        organism = line[5]
        mapping_dict[new_id] = organism
        for old_id in line[-1].split(','):
            mapping_dict[old_id] = organism

    return mapping_dict

mapping_dict = create_mapping_dict(mapping_file_name)


#Define the virus proteins
unique_virus_proteins = {}
unique_host_proteins = {}
true_interactions=0
neg_interactions=0
sars_human_interactions = 0
sars_prots = []
human_prots = []
spike_pos_interactions = 0
spike_neg_interactions=0

#Loop through the proteins and add them
for sample in data:
    
    p1_id = sample['protein_1']['id']
    p2_id = sample['protein_2']['id']
    if p1_id not in unique_virus_proteins:
        unique_virus_proteins[p1_id] = True
        if 'SARS-CoV' in mapping_dict[p1_id]:
            sars_prots.append(p1_id)

    if p2_id not in unique_host_proteins:
        unique_host_proteins[p2_id] = True
        if 'Human' in mapping_dict[p2_id]:
            human_prots.append(p1_id)

    if p1_id == 'P59594': #SPIKE
        if sample['is_interaction'] == 1:
            spike_pos_interactions +=1
        else:
            spike_neg_interactions+=1
        if p2_id == 'Q9BYF1': #ace2
            print('ACE2')
    
    if sample['is_interaction'] == 1:
        true_interactions+=1
        if 'Human' in mapping_dict[p2_id] and 'SARS-CoV' in mapping_dict[p1_id]:
            sars_human_interactions+=1
    else:
        neg_interactions+=1

#Print out relevant data
print('Unique Virus Proteins: {}'.format(len(unique_virus_proteins)))
print('Unique Host Proteins: {}'.format(len(unique_host_proteins)))
print('Pos Interactions: {}'.format(true_interactions))
print('Neg Interactions: {}'.format(neg_interactions))
print('SARS-CoV--Human Interactions: {}'.format(sars_human_interactions))
print('Spike Interactions: {}'.format(spike_pos_interactions))
print('Spike Non-Interactions: {}'.format(spike_neg_interactions))
print('SARS-CoV Proteins: {}'.format(len(sars_prots)))
print('Human Proteins: {}'.format(len(human_prots)))
# stop()