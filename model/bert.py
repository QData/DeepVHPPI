import torch
import torch.nn as nn
from .embedding import get_embedding
from .layers import TransformerEncoderLayer
from pdb import set_trace as stop


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12,dropout=0.1,max_len=512,emb_type='lookup',activation='relu'):
        super().__init__()
        self.hidden = hidden
        self.emb_type = emb_type

        self.embedding = get_embedding(emb_type,vocab_size,hidden,max_len)

        self.transformer_blocks = nn.ModuleList([TransformerEncoderLayer(hidden, attn_heads, dim_feedforward=hidden*4, dropout=dropout, activation=activation,eps=1e-12) for _ in range(n_layers)])

    def forward(self, x_in, mask_in=None):
        x,conv_result = self.embedding(x_in,mask_in)

        if type(x_in) is tuple:
            mask = (torch.cat((x_in[0],torch.ones(x_in[0].size(0),1).cuda().long(),x_in[1]),1) == 0)
        elif mask_in is None:
            mask = (x_in == 0)
        else:
            mask = (mask_in == 0)

        x = x.transpose(0,1)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, src_key_padding_mask=mask)
        x = x.transpose(0,1)

        return x,conv_result
