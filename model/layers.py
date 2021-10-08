import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
# from .attention import MultiheadAttention

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return F.tanh

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    """ Transformer Base Layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,activation="relu",eps=1e-05):
        super(TransformerEncoderLayer, self).__init__()
        print('torch')
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # print('fb')
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=eps) 
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) 

        self.activation = _get_activation_fn(activation) 

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Highway(torch.nn.Module):
    """ Adopted from the AllenNLP/FAIR"""
    
    def __init__(self, input_dim: int, num_layers: int = 1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(num_layers)])
        # self.activation = nn.ReLU()
        self.activation = nn.GELU()

    def forward(self,x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x
