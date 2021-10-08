from torch.utils.data import Dataset
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
import tqdm
import torch
import random
from pdb import set_trace as stop
import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader
from Bio import Seq, SeqIO

def load_starr2020(max_len=-1):
    """Loads the starr2020cov2 under sarscov2, labels the fields, and stores the information
    in the seqs_fitness dictionary, key is a tuple of mutant sequence and strain, and the value
    contains more information, such as fitness, preferences, and wildtype.
    :param max_len: Takes in the maximum number of entries read and stored in the field variable
    """
    strain = 'sars_cov_2'
    wt_seq = SeqIO.read('data/data/sarscov2/data/cov/cov2_spike_wt.fasta', 'fasta').seq

    seqs_fitness = {}
    with open('data/data/sarscov2/data/cov/starr2020cov2/binding_Kds.csv') as f:
        f.readline()
        k = 0
        for line in f:
            k+=1
            if k == max_len:
                break
            fields = line.replace('"', '').rstrip().split(',')
            if fields[5] == 'NA':
                continue
            bert_label_continuous = float(fields[5])
            mutants = fields[-2].split()
            mutable = [ aa for aa in wt_seq ]
            mut_pos = []
            for mutant in mutants:
                orig, mut = mutant[0], mutant[-1]
                pos = int(mutant[1:-1]) - 1 + 330
                assert(wt_seq[pos] == orig)
                mutable[pos] = mut
                mut_pos.append(pos)
            mut_seq = ''.join(mutable)

            if (mut_seq, strain) not in seqs_fitness:
                seqs_fitness[(mut_seq, strain)] = [ {
                    'mut_seq' : mut_seq,
                    'strain': strain,
                    'fitnesses': [ bert_label_continuous ],
                    'preferences': [ bert_label_continuous ],
                    'wildtype': wt_seq,
                    'mut_pos': mut_pos,
                } ]
            else:
                seqs_fitness[(mut_seq, strain)][0][
                    'fitnesses'].append(bert_label_continuous)
                seqs_fitness[(mut_seq, strain)][0][
                    'preferences'].append(bert_label_continuous)

    for fit_key in seqs_fitness:
        seqs_fitness[fit_key][0]['fitness'] = np.median(
            seqs_fitness[fit_key][0]['fitnesses']
        )
        seqs_fitness[fit_key][0]['preference'] = np.median(
            seqs_fitness[fit_key][0]['preferences']
        )

    print(len(seqs_fitness))

    return { strain: wt_seq }, seqs_fitness


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None):
    """Creates the batch sizes and shape for the sequences
    :param sequences: The sequence of values that needs to be shaped
    :param constant_value: Input to the torch and np full functions
    :param dtype: Data type for storing the sequence values
    """
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def merge(list1, list2):
    """Merges two lists
    :param list1: The first list to merge
    :param list2: The second list to merge
    """
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 

class JsonDatasetLM(Dataset):
    """Json Dataset class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab, seq_len, corpus_lines=None):
        """Constructor Method
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.corpus_path = corpus_path


        with open(corpus_path) as f:

            if corpus_lines is not None:
                self.lines = json.load(f)
                random.seed(17)
                random.shuffle(self.lines)
                self.lines = self.lines[0:corpus_lines]
            else:
                self.lines = json.load(f)
            self.corpus_lines = len(self.lines)

            
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq = self.lines[item]['primary']
        family = self.lines[item]['family']
        prot_id = self.lines[item]['id']

        seq,labels = self.process_seq(seq)
        if 'ace2_interaction' in self.lines[item]:
            ace2_interaction = self.lines[item]['ace2_interaction']
        else:
            ace2_interaction = -1

        # random subsequence of lenght self.seq_len
        start_max = max(0,len(seq)-self.seq_len)
        start_val = random.randint(0, start_max)
        end_val = start_val+self.seq_len
        
        bert_input = np.array((seq)[start_val:end_val])
        bert_label = np.array((labels)[start_val:end_val])
        line_len = len(bert_input)
        # stop()
        return bert_input,bert_label,line_len,ace2_interaction
    
    def collate_fn(self, batch):
        bert_input,bert_label,line_len,ace2_interaction = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        bert_label = torch.from_numpy(pad_sequences(bert_label, -1))
        ace2_interaction = torch.tensor([item for item in ace2_interaction])
        line_len = torch.tensor([item for item in line_len])
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label.long(),
                  "bert_evo": torch.tensor([]),
                  "line_len": line_len,
                  "ace2_interaction": ace2_interaction,
                 } 

        return output

    def process_seq(self, sentence):
        tokens = list(sentence)
        output_label = [-1]*len(tokens)
        # random.seed(42)
        for i, token in enumerate(tokens):
            token_idx = self.vocab.stoi.get(token, self.vocab.unk_index)
            prob = random.random()
            # if prob < 0.05:
            #     output_label[i] = token_idx-3
            #     tokens[i] = self.vocab.mask_index

            if prob < 0.15:
                output_label[i] = token_idx-3
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% randomly change token to random token
                elif prob < 0.9: 
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% randomly change token to current token
                else:
                    tokens[i] = token_idx

            else:
                tokens[i] = token_idx
            

        tokens = [self.vocab.cls_index] + tokens
        output_label = [-1] + output_label
        
        return tokens, output_label


class Starr2020Dataset(Dataset):
    """Starr2020 Datset class, used for processing Starr2020 sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab, seq_len, corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        self.corpus_path = corpus_path

        wt_seqs, seqs_fitness = load_starr2020(corpus_lines)

        self.lines = list(seqs_fitness.values())
        
        strains = sorted(wt_seqs.keys())
        
        self.corpus_lines = len(self.lines)

            
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        mut_seq = self.lines[item][0]['mut_seq']
        wt_seq = str(self.lines[item][0]['wildtype'])
        strain = self.lines[item][0]['strain']
        mut_pos = self.lines[item][0]['mut_pos']
        preferences = self.lines[item][0]['preferences']
        fitnesses = self.lines[item][0]['fitnesses']
        preference = self.lines[item][0]['preference']
        fitness = self.lines[item][0]['fitness']

        mut_seq,mut_label = self.process_seq(mut_seq)
        wt_seq,wt_label = self.process_seq(wt_seq)
        
        # random subsequence of lenght self.seq_len
        # start_max = max(0,len(mut_seq)-self.seq_len)
        # start_val = random.randint(0, start_max)
        # end_val = start_val+self.seq_len
        start_val = 0
        end_val = self.seq_len
        
        mut_seq = np.array((mut_seq)[start_val:end_val])
        wt_seq = np.array((wt_seq)[start_val:end_val])
        line_len = len(mut_seq)
        return wt_seq,mut_seq,mut_pos,preference
    
    def collate_fn(self, batch):
        wt_seq,mut_seq,mut_pos,preference = tuple(zip(*batch))
        wt_seq = torch.from_numpy(pad_sequences(wt_seq, self.vocab.pad_index))
        mut_seq = torch.from_numpy(pad_sequences(mut_seq, self.vocab.pad_index))
        preference = torch.tensor([item for item in preference])
        
        output = {"wt_seq": wt_seq,
                  "mut_seq": mut_seq,
                  "mut_pos":mut_pos,
                  "preference": preference,
                 } 

        return output

    def process_seq(self, sentence):
        tokens = list(sentence)
        output_label = [-1]*len(tokens)
        # random.seed(42)
        for i, token in enumerate(tokens):
            token_idx = self.vocab.stoi.get(token, self.vocab.unk_index)
            tokens[i] = token_idx
            
        tokens = [self.vocab.cls_index] + tokens
        output_label = [-1] + output_label
        
        return tokens, output_label

class JsonDatasetTagging(Dataset):
    """Tagging Datset class, used for processing Tagging sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab, seq_len, label_name, corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.label_name = label_name

        with open(corpus_path) as f:
            self.lines = json.load(f)
            if corpus_lines is not None:
                self.lines = self.lines[0:corpus_lines]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq,labels,sample_id,psi_mat = self.get_corpus_line(item)
        seq,labels,psi_mat = self.process_lines(seq,labels,psi_mat)

        bert_input = np.array((seq)[:self.seq_len])
        bert_label = np.array((labels)[:self.seq_len])
        if psi_mat is not None:
            psi_mat = np.array(psi_mat)[:self.seq_len] 
        line_len = len(bert_input)

        assert bert_input.shape[0] == bert_label.shape[0]

        return bert_input,bert_label,psi_mat,line_len,sample_id
    
    def get_corpus_line(self, item):
        primary = self.lines[item]['primary']
        label = self.lines[item][self.label_name]
        if 'hhblits' in self.lines[item]:
            evo = self.lines[item]['hhblits']
        elif 'evolutionary' in self.lines[item]:
            evo = self.lines[item]['evolutionary']
        else:
            evo = None
        
        if 'id' in self.lines[item]:
            sample_id = self.lines[item]['id']
        else:
            sample_id = None
        
        return primary,label,sample_id,evo

    
    def process_lines(self, sentence,labels,psi_mat):
        # print(sentence)
        tokens = list(sentence)
        labels = list(labels)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
        tokens = [self.vocab.cls_index] + tokens
        labels = [-1] + labels

        if psi_mat is not None:
            psi_mat = ([[0]*len(psi_mat[0])]) + psi_mat
        
        return tokens,labels,psi_mat
    
    def collate_fn(self, batch):
        bert_input,bert_label,bert_evo,line_len,sample_id = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        bert_label = torch.from_numpy(pad_sequences(bert_label, -1))
        if bert_evo[0] is not None:
            bert_evo = torch.from_numpy(pad_sequences(bert_evo, self.vocab.pad_index))
        line_len = torch.tensor([item for item in line_len])
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label.long(),
                  "bert_evo": bert_evo,
                  "line_len": line_len,
                  "ids": sample_id,
                 } 

        return output


class JsonDatasetTransmembrane(Dataset):
    """JsonDataset Transmembrane class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab, seq_len, corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path

        with open(corpus_path) as f:
            self.lines = json.load(f)
            if corpus_lines is not None:
                self.lines = self.lines[0:corpus_lines]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq,labels,psi_mat = self.get_corpus_line(item)
        seq,labels,psi_mat = self.process_lines(seq,labels,psi_mat)

        bert_input = np.array((seq)[:self.seq_len])
        bert_label = np.array((labels)[:self.seq_len])

        if psi_mat is not None:
            psi_mat = np.array(psi_mat)[:self.seq_len]
            assert bert_input.shape[0] == psi_mat.shape[0] == bert_label.shape[0]

        line_len = len(bert_input)

        return bert_input,bert_label,psi_mat,line_len
    
    def get_corpus_line(self, item):
        if 'hhblits' in self.lines[item]:
            return self.lines[item]['primary'], self.lines[item]['transmembrane'], self.lines[item]['hhblits']
        else:
            return self.lines[item]['primary'], self.lines[item]['transmembrane'], None
    
    def process_lines(self, sentence,labels,psi_mat):
        # print(sentence)
        tokens = list(sentence)
        labels = list(labels)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
        tokens = [self.vocab.cls_index] + tokens
        labels = [-1] + labels

        if psi_mat is not None:
            psi_mat = ([[0]*len(psi_mat[0])]) + psi_mat
        return tokens,labels,psi_mat
    
    def collate_fn(self, batch):
        bert_input,bert_label,bert_evo,line_len = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        bert_label = torch.from_numpy(pad_sequences(bert_label, -1))
        if bert_evo[0] is not None:
            bert_evo = torch.from_numpy(pad_sequences(bert_evo, self.vocab.pad_index))
        else:
            bert_evo = torch.Tensor([])
        line_len = torch.tensor([item for item in line_len])
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label.long(),
                  "bert_evo": bert_evo,
                  "line_len": line_len
                 } 

        return output

class JsonDataset4Prot(Dataset):
    """JsonDataset 4Prot class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab, seq_len, corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path

        with open(corpus_path) as f:
            self.lines = json.load(f)
            self.corpus_lines = len(self.lines)

            
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq,labels,psi_mat = self.get_corpus_line(item)
        seq,labels,psi_mat = self.process_lines(seq,labels,psi_mat)

        bert_input = np.array((seq)[:self.seq_len])
        bert_label = np.array((labels)[:self.seq_len])
        bert_evo = np.array(psi_mat)[:self.seq_len]
        line_len = len(bert_input)

        return bert_input,bert_label,bert_evo,line_len
    
    def collate_fn(self, batch):
        bert_input,bert_label,bert_evo,line_len = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        bert_label = torch.from_numpy(pad_sequences(bert_label, -1))
        # print(bert_evo)
        # print(type(bert_evo))
        bert_evo = torch.from_numpy(pad_sequences(bert_evo, self.vocab.pad_index))
        line_len = torch.tensor([item for item in line_len])
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label.long(),
                  "bert_evo": bert_evo,
                  "line_len": line_len
                 } 

        return output

    def process_lines(self, sentence,labels,psi_mat):
        # print(sentence)
        tokens = list(sentence)
        labels = list(labels)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        tokens = [self.vocab.cls_index]  + tokens
        labels = [-1] + labels

        psi_mat = ([[0]*len(psi_mat[0])]) + psi_mat
        return tokens,labels,psi_mat

    def get_corpus_line(self, item):
        if 'hhblits' in self.lines[item]:
            return self.lines[item]['primary'], self.lines[item]['ssp'], self.lines[item]['hhblits']
        else:
            return self.lines[item]['primary'], self.lines[item]['ssp'], self.lines[item]['evolutionary']

class JsonDatasetContact(Dataset):
    """JsonDataset Contact class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab, seq_len, corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path

        with open(corpus_path) as f:
            self.lines = json.load(f)
            if corpus_lines is not None:
                self.lines = self.lines[0:corpus_lines]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq,labels,psi_mat = self.get_corpus_line(item)
        seq,labels = self.process_lines(seq,labels)
        bert_input = np.array((seq)[:self.seq_len])
        bert_label = np.array((labels)[:self.seq_len,:self.seq_len])
        bert_evo = np.array(psi_mat)[:self.seq_len]
        line_len = len(bert_input)
        return bert_input,bert_label,bert_evo,line_len
    
    def collate_fn(self, batch):
        bert_input,bert_label,bert_evo,line_len = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        bert_label = torch.from_numpy(pad_sequences(bert_label, self.vocab.pad_index))
        bert_evo = torch.from_numpy(pad_sequences(bert_evo, self.vocab.pad_index))
        line_len = torch.tensor([item for item in line_len])
        
        bert_label = bert_label-1
        output = {"bert_input": bert_input,
                  "bert_label": bert_label.long(),
                  "bert_evo": bert_evo,
                  "line_len": line_len
                 } 

        return output

    def process_lines(self, sentence,labels):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
        labels = np.less(squareform(pdist(np.array(labels))), 8.0).astype(np.int64) 
        labels += 1
        return tokens,labels

    def get_corpus_line(self, item):
        if 'hhblits' in self.lines[item]:
            return self.lines[item]['primary'], self.lines[item]['tertiary'], self.lines[item]['hhblits']
        else:
            return self.lines[item]['primary'], self.lines[item]['tertiary'], self.lines[item]['evolutionary']



class JsonDatasetPairwiseClassification(Dataset):
    """JsonDataset Pairwise class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path, vocab,esm,esm_alphabet,esm_batch_converter, seq_len, corpus_lines=None,random_subseq=False):
        self.vocab = vocab
        self.esm = esm
        self.esm_alphabet = esm_alphabet
        self.esm_batch_converter = esm_batch_converter
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.random_subseq = random_subseq

        with open(corpus_path) as f:
            if corpus_lines is not None:
                self.lines = json.load(f)[0:corpus_lines]
            else:
                self.lines = json.load(f)
            self.corpus_lines = len(self.lines)
        
        
        # self.new_lines = []
        # for line in self.lines:
        #     if line['protein_1']['id'] == 'P59594':
        #         self.new_lines.append(line)
        # self.lines = self.new_lines
        # self.corpus_lines = len(self.lines)


    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq1,id1,seq2,id2,label,bert_label_continuous = self.get_corpus_line(item)

        

        if self.esm:
            sep = self.esm_alphabet.sep_idx
            line_len = len(seq1)+len(seq2)
            return seq1,id1,seq2,id2,label,bert_label_continuous,line_len,sep
        else:
            sep=torch.Tensor()

        
        seq1 = self.process_seq(seq1)
        seq2 = self.process_seq(seq2)


        if self.random_subseq:
            start_max = max(0,len(seq1)-self.seq_len)
            start_val = random.randint(0, start_max)
        else:
            start_val = 0

        end_val = start_val+self.seq_len
    
        seq1 = np.array((seq1)[start_val:end_val])

        if self.random_subseq:
            start_max = max(0,len(seq2)-self.seq_len)
            start_val = random.randint(0, start_max)
        else:
            start_val = 0

        end_val = start_val+self.seq_len
        seq2 = np.array((seq2)[start_val:end_val])
        line_len = len(seq1)+len(seq2)
        # stop()
        return seq1,id1,seq2,id2,label,bert_label_continuous,line_len,sep
    
    def collate_fn(self, batch):

        seq1,id1,seq2,id2,label,bert_label_continuous,line_len,sep = tuple(zip(*batch))

        # stop()
        try:
            bert_label_continuous_out = torch.tensor([item for item in bert_label_continuous])
        except:
            bert_label_continuous_out = None

        if self.esm:
            p1_data = merge(id1,seq1)
            p2_data = merge(id2,seq2)
            s1_labels,s1_seqs,s1_toks = self.esm_batch_converter(p1_data)
            s2_labels,s2_seqs,s2_toks = self.esm_batch_converter(p2_data)

            # Clip Max Len
            s1_toks = s1_toks[:,0:self.seq_len]
            s2_toks = s2_toks[:,0:self.seq_len]

            bert_label = torch.tensor([item for item in label])
            line_len = torch.tensor([item for item in line_len])
            sep = torch.tensor([item for item in sep])
            output = {"bert_input": (s1_toks,s2_toks,sep),
                    "input_id": (list(id1),list(id2)),
                    "bert_label_continuous": bert_label_continuous_out,
                    "bert_label": bert_label,
                    "bert_evo": torch.Tensor([]),
                    "line_len": line_len
                    } 
            
            return output

        else:

            seq1 = torch.from_numpy(pad_sequences(seq1, self.vocab.pad_index))
            seq2 = torch.from_numpy(pad_sequences(seq2, self.vocab.pad_index))
            # print(label)
            bert_label = torch.tensor([item for item in label])
            line_len = torch.tensor([item for item in line_len])
            output = {"bert_input": (seq1,seq2,torch.Tensor()),
                    "input_id": (list(id1),list(id2)),
                    "bert_label": bert_label,
                    "bert_label_continuous": bert_label_continuous,
                    "bert_evo": torch.Tensor([]),
                    "line_len": line_len
                    } 

            return output

    def process_seq(self, sentence):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        tokens = [self.vocab.cls_index] + tokens
        return tokens

    def get_corpus_line(self, item):

        s1 = self.lines[item]['protein_1']['primary']
        id1 = self.lines[item]['protein_1']['id']
        s2 = self.lines[item]['protein_2']['primary']
        id2 = self.lines[item]['protein_2']['id']
        label = self.lines[item]['is_interaction']
        if 'log10ka' in self.lines[item]:
            bert_label_continuous = self.lines[item]['log10ka']
        else:
            bert_label_continuous=-1
        return s1,id1,s2,id2,label,bert_label_continuous
   

class JsonDatasetClassification(Dataset): 
    """JsonDataset Classification class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path,vocab,seq_len,target_name,corpus_lines=None,evo=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.target_name = target_name
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.use_evo = evo

        with open(corpus_path) as f:
            self.lines = json.load(f)
            if corpus_lines is not None:
                self.lines = self.lines[0:corpus_lines]
            self.corpus_lines = len(self.lines)
            
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq,labels,psi_mat = self.get_corpus_line(item)
        seq,labels,psi_mat = self.process_lines(seq,labels,psi_mat)

        bert_input = np.array((seq)[:self.seq_len])
        bert_label = labels
        if self.use_evo:
            bert_evo = np.array(psi_mat)[:self.seq_len]
        else:
            bert_evo = None
        line_len = len(bert_input)

        return bert_input,bert_label,bert_evo,line_len
    
    def collate_fn(self, batch):
        bert_input,bert_label,bert_evo,line_len = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        # bert_label = torch.from_numpy(pad_sequences(bert_label, self.vocab.pad_index))
        bert_label = torch.tensor([item for item in bert_label])
        if self.use_evo:
            bert_evo = torch.from_numpy(pad_sequences(bert_evo, self.vocab.pad_index))
        else:
            bert_evo = torch.Tensor([])
        line_len = torch.tensor([item for item in line_len])
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label.long(),
                  "bert_evo": bert_evo,
                  "line_len": line_len
                 } 

        return output
 
    def process_lines(self, sentence,labels,psi_mat):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        tokens = [self.vocab.cls_index] + tokens
        if self.use_evo:
            psi_mat = ([[0]*len(psi_mat[0])]) + psi_mat
        return tokens,labels,psi_mat

    def get_corpus_line(self, item):
        if 'hhblits' in self.lines[item]:
            # return self.lines[item]['primary'], self.lines[item][self.target_name], self.lines[item]['evolutionary']
            return self.lines[item]['primary'], self.lines[item][self.target_name], self.lines[item]['hhblits']
        else:
            return self.lines[item]['primary'], self.lines[item][self.target_name], None



class JsonDatasetRegression(Dataset): 
    """JsonDataset Regression class, used for processing sequences
    :param Dataset: Takes in an object with the data
    """
    def __init__(self, corpus_path,vocab,seq_len,output_var,corpus_lines=None,evo=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.output_var = output_var

        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.use_evo = evo

        with open(corpus_path) as f:
            self.lines = json.load(f)
            if corpus_lines is not None:
                self.lines = self.lines[0:corpus_lines]
            self.corpus_lines = len(self.lines)

            
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        seq,labels,psi_mat = self.get_corpus_line(item)
        seq,labels,psi_mat = self.process_lines(seq,labels,psi_mat)

        bert_input = np.array((seq)[:self.seq_len])
        if psi_mat is not None:
            psi_mat = np.array(psi_mat)[:self.seq_len]
        
        bert_label = labels
        
        line_len = len(bert_input)

        return bert_input,bert_label,psi_mat,line_len
    
    def collate_fn(self, batch):
        bert_input,bert_label,bert_evo,line_len = tuple(zip(*batch))
        bert_input = torch.from_numpy(pad_sequences(bert_input, self.vocab.pad_index))
        bert_label = torch.tensor([item for item in bert_label])
        if bert_evo[0] is not None:
            bert_evo = torch.from_numpy(pad_sequences(bert_evo, self.vocab.pad_index))
        else:
            bert_evo = torch.Tensor([])
        
        line_len = torch.tensor([item for item in line_len])
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "bert_evo": bert_evo,
                  "line_len": line_len,
                 } 

        return output
 
    def process_lines(self, sentence,labels,psi_mat):
        tokens = list(sentence)
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
        if psi_mat is not None:
            psi_mat = ([[0]*len(psi_mat[0])]) + psi_mat
        tokens = [self.vocab.cls_index] + tokens
        return tokens,labels,psi_mat

    def get_corpus_line(self, item):
        if self.use_evo:
            return self.lines[item]['primary'], self.lines[item][self.output_var], self.lines[item]['hhblits']
        else:
            return self.lines[item]['primary'], self.lines[item][self.output_var], None


def load_data_single(args,vocab,esm_alphabet,esm_batch_converter):
    """Load method for each of the dataets
    :param args: arguments from command line
    :param vocab: vocabulary from dataset
    :param esm_alphabet: protein model alphabet 
    :param esm_alphabet: protein model batch converter 
    """
    if args.task == 'secondary':
        num_classes =  3
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetTagging(args.train_dataset,vocab,args.seq_len,'ss3',corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = JsonDatasetTagging(args.valid_dataset,vocab,args.seq_len,'ss3') if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetTagging(args.test_dataset,vocab,args.seq_len,'ss3') if args.test_dataset is not None else None

    if args.task == 'transmembrane':
        num_classes =  4
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetTransmembrane(args.train_dataset,vocab,args.seq_len,corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = JsonDatasetTransmembrane(args.valid_dataset,vocab,args.seq_len) if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetTransmembrane(args.test_dataset,vocab,args.seq_len) if args.test_dataset is not None else None

    elif args.task == 'contact':
        num_classes = 1
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetContact(args.train_dataset, vocab, args.seq_len,corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset)
        valid_dataset = JsonDatasetContact(args.valid_dataset, vocab, args.seq_len) if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetContact(args.test_dataset, vocab, args.seq_len) if args.test_dataset is not None else None

        
    elif args.task == 'homology':
        num_classes = 1195
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetClassification(args.train_dataset,vocab,args.seq_len,'fold_label',corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset)
        valid_dataset = JsonDatasetClassification(args.valid_dataset,vocab,args.seq_len,'fold_label') if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetClassification(args.test_dataset,vocab,args.seq_len,'fold_label') if args.test_dataset is not None else None

    elif args.task == 'fluorescence':
        num_classes = 1
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetRegression(args.train_dataset, vocab,args.seq_len, 'log_fluorescence',corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset)
        valid_dataset = JsonDatasetRegression(args.valid_dataset, vocab, args.seq_len,'log_fluorescence') if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetRegression(args.test_dataset, vocab,args.seq_len,'log_fluorescence') if args.test_dataset is not None else None

    
    elif args.task == 'stability':
        num_classes = 1
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetRegression(args.train_dataset,vocab,args.seq_len,'stability',corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset)
        valid_dataset = JsonDatasetRegression(args.valid_dataset,vocab,args.seq_len, 'stability') if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetRegression(args.test_dataset, vocab,args.seq_len, 'stability') if args.test_dataset is not None else None

    
    elif args.task == 'solubility':
        num_classes = 2
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetClassification(args.train_dataset, vocab,args.seq_len,'solubility',corpus_lines=args.corpus_lines,evo=True) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset)
        valid_dataset = JsonDatasetClassification(args.valid_dataset, vocab,args.seq_len,'solubility',evo=True) if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetClassification(args.test_dataset, vocab,args.seq_len,'solubility',evo=True) if args.test_dataset is not None else None

    
    elif args.task == 'localization':
        num_classes = 24
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetClassification(args.train_dataset, vocab,args.seq_len,'localization',corpus_lines=args.corpus_lines,evo=True) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset)
        valid_dataset = JsonDatasetClassification(args.valid_dataset, vocab,args.seq_len,'localization',evo=True) if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetClassification(args.test_dataset, vocab,args.seq_len,'localization',evo=True) if args.test_dataset is not None else None

    elif args.task == '4prot':
        num_classes =  3
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDataset4Prot(args.train_dataset, vocab, seq_len=args.seq_len,corpus_lines=args.corpus_lines) if args.train_dataset is not None else None
        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = JsonDataset4Prot(args.valid_dataset, vocab, seq_len=args.seq_len) if args.valid_dataset is not None else None
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDataset4Prot(args.test_dataset, vocab, seq_len=args.seq_len) if args.test_dataset is not None else None


    elif args.task in ['biogrid','ppi']:
        num_classes =  1
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetPairwiseClassification(args.train_dataset, vocab, args.esm, esm_alphabet,esm_batch_converter, seq_len=args.seq_len,corpus_lines=args.corpus_lines,random_subseq=True) if args.train_dataset is not None else None
        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = JsonDatasetPairwiseClassification(args.valid_dataset, vocab,args.esm, esm_alphabet,esm_batch_converter, seq_len=args.seq_len2,corpus_lines=args.corpus_lines) if args.valid_dataset is not None else None
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetPairwiseClassification(args.test_dataset, vocab,args.esm, esm_alphabet,esm_batch_converter,seq_len=args.seq_len2,corpus_lines=args.corpus_lines) if args.test_dataset is not None else None

    elif args.task == 'malaria':
        num_classes =  2
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetTagging(args.train_dataset,vocab,args.seq_len,'label',corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = JsonDatasetTagging(args.valid_dataset,vocab,args.seq_len,'label') if args.valid_dataset is not None else None

        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetTagging(args.test_dataset,vocab,args.seq_len,'label') if args.test_dataset is not None else None
    
    elif args.task == 'lm':
        num_classes = len(vocab)-3
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = JsonDatasetLM(args.train_dataset, vocab, seq_len=args.seq_len,corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = JsonDatasetLM(args.valid_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines) if args.valid_dataset is not None else None
        
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = JsonDatasetLM(args.test_dataset, vocab, seq_len=args.seq_len,corpus_lines=args.corpus_lines) if args.test_dataset is not None else None

    elif args.task == 'test_lm':
        num_classes = len(vocab)-3
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = Starr2020Dataset(args.train_dataset, vocab, seq_len=args.seq_len,corpus_lines=args.corpus_lines) if args.train_dataset is not None else None

        print("Loading Valid Dataset", args.valid_dataset) 
        valid_dataset = Starr2020Dataset(args.valid_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines) if args.valid_dataset is not None else None
        
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = Starr2020Dataset(args.test_dataset, vocab, seq_len=args.seq_len,corpus_lines=args.corpus_lines) if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=args.num_workers) if train_dataset is not None else None

    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size,shuffle=False,collate_fn=valid_dataset.collate_fn, num_workers=args.num_workers) if valid_dataset is not None else None

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn,num_workers=args.num_workers) if test_dataset is not None else None

    data = {
        'train':train_data_loader,
        'valid':valid_data_loader,
        'test':test_data_loader,
        'num_classes':num_classes
    }

    return data


def get_data_loader(task,train_dataset,valid_dataset,test_dataset,num_classes,args):
    """Get data given the task, training, validation, and test dataset
    :param task: could be secondary, 
    :param train_dataset: data to train the model with
    :param valid_dataset: data to validate the model with
    :param test_dataset: data to test the model with
    :param num_classes: number of classes of data to input
    :param args: arguments specified when running such as batch_size
    """
    train_data_loader=DataLoader(train_dataset,args.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)
    valid_data_loader=DataLoader(valid_dataset,args.batch_size,shuffle=False,collate_fn=valid_dataset.collate_fn, num_workers=args.num_workers) if valid_dataset is not None else None
    test_data_loader=DataLoader(test_dataset,args.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn,num_workers=args.num_workers) if test_dataset is not None else None
    

    data = {
        'task':task,
        'train':train_data_loader,
        'valid':valid_data_loader,
        'test':test_data_loader,
        'num_classes':num_classes
    }
    return data

def load_data_multi(args,vocab):
    """Get data for multiple tasks
    :param args: arguments specified when running such as seq_len and corpus_lines
    :param vocab: vocab for a given dataset used as an input for the dataset
    """
    data = []

    num_classes =  3
    task = 'secondary'
    train_dataset = JsonDatasetTagging('data/data/secondary/train.json',vocab,args.seq_len,'ss3',corpus_lines=args.corpus_lines)
    valid_dataset = JsonDatasetTagging('data/data/secondary/valid.json',vocab,args.seq_len,'ss3',corpus_lines=args.corpus_lines)
    test_dataset = JsonDatasetTagging('data/data/secondary/cb513.json',vocab,args.seq_len,'ss3',corpus_lines=args.corpus_lines)
    data.append(get_data_loader(task,train_dataset,valid_dataset,test_dataset,num_classes,args))

    num_classes = 1
    task = 'contact'
    train_dataset = JsonDatasetContact('data/data/contact/test.json', vocab, args.seq_len,corpus_lines=args.corpus_lines)
    valid_dataset = JsonDatasetContact('data/data/contact/test.json', vocab, args.seq_len,corpus_lines=args.corpus_lines)
    test_dataset = JsonDatasetContact('data/data/contact/test.json', vocab, args.seq_len,corpus_lines=args.corpus_lines)
    data.append(get_data_loader(task,train_dataset,valid_dataset,test_dataset,num_classes,args))

    num_classes =  1195
    task = 'homology' 
    train_dataset = JsonDatasetClassification('data/data/homology/train.json',vocab,args.seq_len,'fold_label',corpus_lines=args.corpus_lines,evo=args.use_evo)
    valid_dataset = JsonDatasetClassification('data/data/homology/valid.json',vocab,args.seq_len,'fold_label',corpus_lines=args.corpus_lines,evo=args.use_evo)
    test_dataset = JsonDatasetClassification('data/data/homology/test.json',vocab,args.seq_len,'fold_label',corpus_lines=args.corpus_lines,evo=args.use_evo)
    data.append(get_data_loader(task,train_dataset,valid_dataset,test_dataset,num_classes,args))

    return data


def load_data(args,vocab,esm_alphabet,esm_batch_converter):
    """General load method for a given dataset
    :param args: arguments including type of task
    :param vocab: vocabulary from dataset
    :param esm_alphabet: protein model alphabet 
    :param esm_alphabet: protein model batch converter 
    """
    if args.task == 'multi':
        return load_data_multi(args,vocab)
    else:
        return load_data_single(args,vocab,esm_alphabet,esm_batch_converter)
