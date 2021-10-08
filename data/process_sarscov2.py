from pdb import set_trace as stop
import json
import numpy as np
import random
from Bio import SeqIO
import copy
ace2_seq='MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSIKVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPRISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPPVSIWLIVFGVVMGVIVVGIVILIFTGIRDRKKKNKARSGENPYASIDISKGENNPGFQNTDDVQTSF'

pairs = []
seqs = []
dissociation_vals = []

patient_seq_fnames = [  'data/sarscov2/data/cov/sars_cov2_seqs.fa',
                        'data/sarscov2/data/cov/viprbrc_db.fasta',
                        'data/sarscov2/data/cov/gisaid.fasta' 
                     ]

def parse_viprbrc(entry):
    fields = entry.split('|')
    if fields[7] == 'NA':
        date = None
    else:
        date = fields[7].split('/')[0]
        date = dparse(date.replace('_', '-'))

    country = fields[9]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[5],
        'host': fields[8],
        'group': species2group[fields[8]],
        'country': country,
        'continent': continent,
        'dataset': 'viprbrc',
    }
    return meta

def parse_nih(entry):
    fields = entry.split('|')

    country = fields[3]
    # from locations import country2continent
    # if country in country2continent:
    #     continent = country2continent[country]
    # else:
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

def parse_gisaid(entry):
    fields = entry.split('|')

    type_id = fields[1].split('/')[1]

    if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                    'pangolin', 'tiger' }:
        host = type_id
        country = 'NA'
        continent = 'NA'
    else:
        host = 'human'
        from locations import country2continent
        if type_id in country2continent:
            country = type_id
            continent = country2continent[country]
        else:
            country = 'NA'
            continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[1],
        'host': host,
        'group': species2group[host].lower(),
        'country': country,
        'continent': continent,
        'dataset': 'gisaid',
    }
    return meta

def process(fnames):
    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 1000:
                continue
            if str(record.seq).count('X') > 0:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            if fname == 'data/cov/viprbrc_db.fasta':
                meta = parse_viprbrc(record.description)
            elif fname == 'data/cov/gisaid.fasta':
                meta = parse_gisaid(record.description)
            else:
                meta = parse_nih(record.description)
            meta['accession'] = record.description
            seqs[record.seq].append(meta)

    return seqs

patient_seqs = process(patient_seq_fnames)

output_list =[]
idx = 1
for key,val in patient_seqs.items():
    # stop()
    sample = {}
    sample['id'] = idx
    sample['primary'] = str(key)
    sample['family'] = ''
    output_list.append(sample)
    idx+=1
    

# stop()
with open('data/sarscov2/data/cov/cov_all.json', 'w') as fout:
    json.dump(output_list , fout)

random.shuffle(output_list)
train = output_list[0:int(0.8*len(output_list))]
test = output_list[int(0.8*len(output_list)):]
with open('data/sarscov2/data/cov/cov_train.json', 'w') as fout:
    json.dump(train , fout)
with open('data/sarscov2/data/cov/cov_test.json', 'w') as fout:
    json.dump(test , fout)
# stop()


AAs = [ 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'Z', 'J', 'U', 'B'] 
vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }


simulated_pairs = []
for seq,seq_info in patient_seqs.items():
    orig_seq=str(seq)
    orig_pair = {"protein_1": {"id": "S", "primary": orig_seq}, "protein_2": {"id": "Q9BYF1", "primary": ace2_seq}, "is_interaction": 1}
    simulated_pairs.append(orig_pair)

    num_neg = 2
    num_mutations = 11
    for _ in range(num_neg):
        sample_mutations = random.randrange(1,num_mutations)

        seq_len = len(orig_seq)

        mut_seq = copy.deepcopy(orig_seq)

        for _ in range(sample_mutations):
            rand_pos = random.randrange(seq_len)
            orig_AA = orig_seq[rand_pos]

            rand_AA_idx = random.randrange(len(AAs))
            mut_AA = AAs[rand_AA_idx]
            
            while mut_AA == orig_AA:
                rand_AA_idx = random.randrange(len(AAs))
                mut_AA = AAs[rand_AA_idx]

            
            mut_seq = mut_seq[:rand_pos] + mut_AA + mut_seq[rand_pos + 1:]

        
        mut_pair = {"protein_1": {"id": "S", "primary": mut_seq}, "protein_2": {"id": "Q9BYF1", "primary": ace2_seq}, "is_interaction": 0}
        

        simulated_pairs.append(mut_pair)
    


with open('data/sarscov2/simulated_pairs_'+str(num_neg)+'neg_'+str(num_mutations)+'mut.json', 'w') as fp:
    json.dump(simulated_pairs, fp)

# exit()

with open('data/sarscov2/cov_mutated_fitness.csv','r') as f:
    next(f)
    for line in f:
        line = line.strip()
        spike_seq = line.split(',')[0]
        # spike_seq = spike_seq[319:542]
        ka = float(line.split(',')[1])
        dissociation_vals.append(ka)
        ka_diff = ka-10.76
        # append = False
        # if ka_diff < 0:
        #     append = True
        #     discrete_diff=0
        # elif ka_diff > 0:
        #     append = True
        #     discrete_diff=1

        
        # if append:
        #     pair = {"protein_1": {"id": "S", "primary": spike_seq}, "protein_2": {"id": "Q9BYF1", "primary": ace2_seq}, "is_interaction": ka}
        #     pairs.append(pair)
        # if ka != 6.0:
        pair = {"protein_1": {"id": "S", "primary": spike_seq}, "protein_2": {"id": "Q9BYF1", "primary": ace2_seq}, "log10ka": ka}
        pairs.append(pair)

        seq = {'id': 'S', 'primary': spike_seq, 'family': '','ace2_interaction': ka}
        seqs.append(seq)

dissociation_vals = np.array(dissociation_vals)
median = np.median(dissociation_vals)
pos_vals = 0
neg_vals = 0
for pair in pairs:
    log10ka_val = pair['log10ka']
    if log10ka_val >= median:
        pair['is_interaction'] = 1
        pos_vals+=1
    else:
        pair['is_interaction'] = 0
        neg_vals+=1

print('Pos: {}'.format(pos_vals))
print('Neg: {}'.format(neg_vals))

hist,edges = np.histogram(dissociation_vals,bins=10)
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
plt.bar(edges[1:],hist)
plt.savefig('histogram.png')


wt_seq = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLT'

pos,neg = 0,0
pos_counts,neg_counts = [],[]
for pair in pairs:
    p1 = pair['protein_1']['primary']
    p2 = pair['protein_2']['primary']
    is_interaction = pair['is_interaction']
    count = sum(1 for a, b in zip(p1, wt_seq) if a != b)
    if is_interaction == 1:
        pos+=1
        pos_counts.append(count)
    else:
        neg+=1
        neg_counts.append(count)

pos_counts = np.array(pos_counts)
neg_counts = np.array(neg_counts)

# stop()

with open('data/sarscov2/mutation_pairs_all.json', 'w') as fp:
    json.dump(pairs, fp)

# with open('data/sarscov2/mutation_seqs.json', 'w') as fp:
    # json.dump(seqs, fp)


for train_percentage in [0.001,0.01,0.1]:
    random.shuffle(pairs)

    train_len = int(len(pairs)*train_percentage)
    val_len = int(len(pairs)*0.1)
    train = pairs[0:train_len]
    valid = pairs[train_len:train_len+val_len]
    test = pairs[train_len+val_len:]

    train_percentage_str = str(train_percentage).replace('.','')
    with open('data/sarscov2/mutation_pairs_train'+train_percentage_str+'.json', 'w') as fp:
        json.dump(train, fp)

    with open('data/sarscov2/mutation_pairs_valid'+train_percentage_str+'.json', 'w') as fp:
        json.dump(valid, fp)

    with open('data/sarscov2/mutation_pairs_test'+train_percentage_str+'.json', 'w') as fp:
        json.dump(test, fp)

# stop()