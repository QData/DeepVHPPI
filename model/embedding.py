import torch.nn as nn
import torch
import math
import numpy as np
from pdb import set_trace as stop
from typing import List, Tuple
import torch.nn.functional as F
from .layers import Highway



class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1,max_len=512,pos='sin'):
        super().__init__()
        self.token = nn.Embedding(vocab_size,embed_size,padding_idx=0)
        self.pos = pos
        if self.pos == 'sin':
            self.position = PositionalEmbedding(d_model=self.token.embedding_dim,max_len=max_len)
        else:
            self.position = nn.Embedding(max_len,embed_size)
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence,psi=None):
        if self.pos == 'sin':
            embeddings = self.token(sequence) + self.position(sequence)
        else:
            position_ids = torch.arange(sequence.size(1), dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).expand_as(sequence)
            embeddings = self.token(sequence) + self.position(position_ids) 

        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
        
        return embeddings,None


class BERTEmbeddingConv(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1,max_len=512,pos='sin'):
        super().__init__()
        highway_layers = 1

        onehot_dim = vocab_size-1
        self.onehot = nn.Embedding(vocab_size,onehot_dim,padding_idx=0)
        self.onehot.weight.requires_grad = False
        self.onehot.weight[1:] = torch.eye(onehot_dim)

        filters = [(1,128),(3,256),(5,384),(7,512),(9,512),(11,512)]
        # filters = [(1,128),(3,256),(5,256),(7,256),(9,256),(11,256),(13,256)]
        # filters = [(1,64),(3,128),(5,128),(7,128),(9,128),(11,128),(13,128)]
        # filters = [(1,128),(2,256),(3,384),(4,512),(5,512),(6,512)]
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(onehot_dim, out_c, kernel_size=width,padding=int(width/2))
            )
        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None
        self.projection = nn.Linear(last_dim, embed_size)

        self.pos = pos
        if self.pos == 'sin':
            self.position = PositionalEmbedding(d_model=embed_size,max_len=max_len)
        else:
            self.position = nn.Embedding(max_len,embed_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12) 
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, sequence,psi=None):
        if self.pos == 'sin':
            
            char_embs = self.onehot(sequence)
            char_embs = char_embs.transpose(1, 2)  # BTC -> BCT


            conv_result = []
            for conv in self.convolutions:

                x_cls = conv(char_embs)[:,:,0:1] # separate the CLS token from the rest of the seq
                x = conv(char_embs)[:,:,1:sequence.size(1)]
                x = torch.cat((x_cls,x),-1)
                x = self.activation(x)
                x = self.dropout(x)
                conv_result.append(x)

            # stop()
            x = torch.cat(conv_result, dim=1).transpose(1,2)
            
            if self.highway is not None:
                x = self.highway(x)

            seq_emb = self.projection(x)

            pos_embeddings = self.position(sequence)
            embeddings = seq_emb + pos_embeddings
        else:
            position_ids = torch.arange(sequence.size(1), dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).expand_as(sequence)
            embeddings = self.token(sequence) + self.position(position_ids) 

        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
        
        return embeddings,conv_result


 

class BERTEmbeddingContinuous(nn.Module):
    '''
    Embedding for continuous inputs
    '''
    def __init__(self, vocab_size, embed_size, dropout=0.1,max_len=512,pos='sin'):
        super().__init__()
        highway_layers = 1

        filters = [(1,128),(3,256),(5,384),(7,512),(9,512),(11,512)]

        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(vocab_size, out_c, kernel_size=width,padding=int(width/2))
            )
        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None
        self.projection = nn.Linear(last_dim, embed_size)

        self.pos = pos
        if self.pos == 'sin':
            self.position = PositionalEmbedding(d_model=embed_size,max_len=max_len)
        else:
            self.position = nn.Embedding(max_len,embed_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12) 
        self.dropout = nn.Dropout(p=dropout)
        self.input_dropout = nn.Dropout(p=0.1)

    def forward(self, sequence,psi=None):
        if self.pos == 'sin':
            
            sequence_t = sequence.transpose(1, 2)
            sequence_t = self.input_dropout(sequence_t)
            conv_result = []
            for conv in self.convolutions:
                x = conv(sequence_t)[:,:,0:sequence_t.size(-1)]
                x = self.activation(x)
                x = self.dropout(x)
                conv_result.append(x)

            x = torch.cat(conv_result, dim=1).transpose(1,2)
            
            if self.highway is not None:
                x = self.highway(x)

            seq_emb = self.projection(x)

            pos_embeddings = self.position(sequence)
            embeddings = seq_emb + pos_embeddings
        else:
            position_ids = torch.arange(sequence.size(1), dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).expand_as(sequence)
            embeddings = self.token(sequence) + self.position(position_ids) 

        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
        
        return embeddings
        


class BERTEmbeddingBoth(nn.Module):
    '''
    Embedding for both token and continuous inputs
    '''
    def __init__(self, vocab_size, embed_size, dropout=0.1,max_len=512,pos='sin'):
        super().__init__()
        highway_layers = 1

        onehot_dim = 29-1
        self.onehot = nn.Embedding(29,onehot_dim,padding_idx=0)
        self.onehot.weight.requires_grad = False
        self.onehot.weight[1:] = torch.eye(onehot_dim)

        conv_dim = onehot_dim + vocab_size
        filters = [(1,128),(3,256),(5,384),(7,512),(9,512),(11,512)]
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(conv_dim, out_c, kernel_size=width,padding=int(width/2))
            )
        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None
        self.projection = nn.Linear(last_dim, embed_size)


        self.position = PositionalEmbedding(d_model=embed_size,max_len=max_len)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12) 
        self.dropout = nn.Dropout(p=dropout)
        self.input_dropout = nn.Dropout(p=0.1)

    def forward(self, cont_sequence,token_sequence):
        # evo_emb = self.evo_projection(cont_sequence)
        # stop()
        char_embs = self.onehot(token_sequence)

        conv_input = torch.cat((char_embs,cont_sequence),2)

        conv_input = conv_input.transpose(1, 2)
        conv_result = []
        for conv in self.convolutions:
            x = conv(conv_input)[:,:,0:token_sequence.size(1)]
            x = self.activation(x)
            x = self.dropout(x)
            conv_result.append(x)
        x = torch.cat(conv_result, dim=1).transpose(1,2)
        if self.highway is not None:
            x = self.highway(x)
        seq_emb = self.projection(x)

        pos_embeddings = self.position(token_sequence)
        embeddings = seq_emb + pos_embeddings #+ evo_emb

        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
        
        return embeddings

class BERTEmbeddingPair(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1,max_len=512,pos='sin'):
        super().__init__()
        highway_layers = 1
        
        self.onehot_dim = vocab_size
        self.onehot = nn.Embedding(vocab_size,self.onehot_dim,padding_idx=0)
        self.onehot.weight.requires_grad = False
        self.onehot.weight.fill_(0)
        self.onehot.weight[1:,0:-1] = torch.eye(self.onehot_dim-1)

        filters = [(1,128),(3,256),(5,384),(7,512),(9,512),(11,512)]
        # filters = [(1,128),(3,256),(5,256),(7,256),(9,256),(11,256),(13,256)]
        # filters = [(1,64),(3,128),(5,128),(7,128),(9,128),(11,128),(13,128)]
        # filters = [(1,128),(2,256),(3,384),(4,512),(5,512),(6,512)]
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(self.onehot_dim, out_c, kernel_size=width,padding=int(width/2))
            )
        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None
        self.projection = nn.Linear(last_dim, embed_size)

        self.pos = pos
        if self.pos == 'sin':
            self.position = PositionalEmbedding(d_model=embed_size,max_len=max_len*2 + 1)
        else:
            self.position = nn.Embedding(max_len,embed_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12) 
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, sequence,psi=None):
        
        seq1 = self.onehot(sequence[0])
        seq2 = self.onehot(sequence[1])
        sep_emb = torch.zeros(seq1.size(0),1,self.onehot_dim).cuda()
        sep_emb[:,:,self.onehot_dim-1] = 1

        char_embs = torch.cat((seq1,sep_emb,seq2),1)
        char_embs = char_embs.transpose(1, 2)  # BTC -> BCT

        conv_result = []
        for conv in self.convolutions:
            x = conv(char_embs)
            # x, _ = torch.max(x, -1)
            x = self.activation(x)
            x = self.dropout(x)
            conv_result.append(x)

        x = torch.cat(conv_result, dim=1).transpose(1,2)
        
        if self.highway is not None:
            x = self.highway(x)

        seq_emb = self.projection(x)

        try:
            pos_embeddings = self.position(char_embs.transpose(1, 2))
        except:
            stop()
            
        embeddings = seq_emb + pos_embeddings

        embeddings = self.layer_norm(embeddings) 
        embeddings = self.dropout(embeddings) 
        
        return embeddings


def get_embedding(emb_type,vocab_size,hidden,max_len):
    if emb_type == 'lookup':
        embedding = BERTEmbedding(vocab_size, hidden,max_len=max_len,pos='sin')
    elif emb_type == 'conv':
        embedding = BERTEmbeddingConv(vocab_size, hidden,max_len=max_len,pos='sin')
    elif emb_type == 'continuous':
        embedding = BERTEmbeddingContinuous(vocab_size, hidden,max_len=max_len,pos='sin')
    elif emb_type == 'both':
        embedding = BERTEmbeddingBoth(vocab_size, hidden,max_len=max_len,pos='sin')
    elif emb_type == 'pair':
        embedding = BERTEmbeddingPair(vocab_size, hidden,max_len=max_len,pos='sin')
    else:
        raise Exception('emb_type was: {}'.format(emb_type))

    return embedding