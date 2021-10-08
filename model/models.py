import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import BERT
from pdb import set_trace as stop
from torch.nn.utils.weight_norm import weight_norm
from .layers import TransformerEncoderLayer
from .embedding import PositionalEmbedding
# !import code; code.interact(local=vars())


class TokenClassifierModel(nn.Module):
    def __init__(self, bert, n_classes, dropout,esm=False):
        super().__init__()
        self.bert = bert
        self.esm = esm
        
            
        if self.esm:
            self.num_esm_layers = len(bert.layers)
            hidden_size = bert.layers[-1].self_attn.out_proj.in_features
        else:
            hidden_size = self.bert.hidden
        
        self.classifier = TokenClassifier(hidden_size,n_classes)

    def forward(self, x_in, psi_mat=None):
        if self.esm:
            x1 = self.bert(x_in, repr_layers=[self.num_esm_layers])
            x  = x1["representations"][self.num_esm_layers]
        else:
            x,conv_out = self.bert(x_in)

        task_outputs = self.classifier.forward(x)

        return task_outputs

class TokenClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()
        # self.classify = nn.Sequential(
        #     nn.Dropout(0.1, inplace=True),
        #     # nn.BatchNorm1d(hidden_size),
        #     weight_norm(nn.Linear(hidden, n_classes), dim=None),
        #     )
        self.dense = nn.Linear(hidden, hidden)
        self.activation_fn = torch.tanh 
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(hidden,eps=1e-12) 
        self.out_proj = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x = self.classify(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x

# class TokenClassifier(nn.Module):
#     def __init__(self,hidden_size: int,num_labels: int,ignore_index: int = -100):
#         super().__init__()
#         self.classify = nn.Sequential(
#             nn.Dropout(0.1, inplace=True),
#             # nn.BatchNorm1d(hidden_size),  # Added this
#             weight_norm(nn.Conv1d(hidden_size, hidden_size, 5, padding=2), dim=None),
#             nn.GELU(),
#             nn.Dropout(0.1, inplace=True),
#             weight_norm(nn.Conv1d(hidden_size, num_labels, 3, padding=1), dim=None))
#         self.num_labels = num_labels
#         self._ignore_index = ignore_index
#     def forward(self, sequence_output):
#         sequence_output = sequence_output.transpose(1, 2)
#         sequence_logits = self.classify(sequence_output)
#         sequence_logits = sequence_logits.transpose(1, 2).contiguous()
#         outputs = sequence_logits
#         return outputs 


class SequenceClassifierModel(nn.Module):
    def __init__(self, bert: BERT, tasks, dropout,esm=False):
        super().__init__()
        self.bert = bert
        self.esm = esm
        if self.esm:
            self.num_esm_layers = len(bert.layers)
            hidden_size = bert.layers[-1].self_attn.out_proj.in_features
        else:
            hidden_size = self.bert.hidden

        self.classifier = SequenceClassifier(hidden_size,tasks)


    def forward(self, x_in, psi_mat):
        if self.esm:
            x1 = self.bert(x_in, repr_layers=[self.num_esm_layers])
            x  = x1["representations"][self.num_esm_layers]
        else:
            x,conv_out = self.bert(x_in)


        task_outputs = self.classifier.forward(x)
        return task_outputs


class SequenceClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden, n_classes)
        
    def forward(self, x):
        x = x[:,0,:].contiguous()
        x = self.linear1(self.dropout(x))
        return x


class PairwiseClassifierModel(nn.Module):
    def __init__(self, bert: BERT, tasks, dropout,seq_len=512,esm=False):
        super().__init__()
        self.bert = bert
        self.esm =  esm
        

        if self.esm:
            self.num_esm_layers = len(bert.layers)
            hidden_size = bert.layers[-1].self_attn.out_proj.in_features
        else:
            hidden_size = self.bert.hidden

        # self.classifier = OrderAgnosticClassifier(hidden_size,tasks)
        self.classifier = OrderedClassifier(hidden_size,tasks,dropout)
        # self.classifier = TransformerClassifier(hidden_size,tasks,seq_len=seq_len)

    def forward(self, x_in, target):

        if self.esm:
            x1out = self.bert(x_in[0], repr_layers=[self.num_esm_layers])
            x2out= self.bert(x_in[1], repr_layers=[self.num_esm_layers])
            x1  = x1out["representations"][self.num_esm_layers]
            x2  = x2out["representations"][self.num_esm_layers]
        else:
            x1,conv_out1 = self.bert(x_in[0])
            x2,conv_out2= self.bert(x_in[1])

        task_outputs = self.classifier.forward(x1,x2,x_in[0],x_in[1])

        return task_outputs

class CatClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden, int(hidden))
        self.nonlinear = nn.GELU()
        self.linear2 = nn.Linear(int(hidden), 1)
    def forward(self, cat_rep):
        x1 = cat_rep[:,0,:].contiguous()
        v_out = self.dropout(x1)
        v_out = self.linear1(v_out)
        v_out = self.nonlinear(v_out)
        v_out = self.dropout(v_out)
        out = self.linear2(v_out)
        return out.view(-1)

class PairwiseClassifierModelNew(nn.Module):
    """
    This is for concatenating two sequences and inputting them both to a single transformer
    """
    def __init__(self, bert: BERT,tasks, dropout,seq_len=512,esm=False):
        super().__init__()
        self.bert = bert
        self.esm =  esm
        
        if self.esm:
            self.num_esm_layers = len(bert.layers)
            hidden_size = bert.layers[-1].self_attn.out_proj.in_features
        else:
            hidden_size = self.bert.hidden

        self.classifier = CatClassifier(hidden_size,tasks,dropout)

    def forward(self, x_in, target):
        xcat = torch.cat((x_in[0],x_in[2].unsqueeze(1),x_in[1]),1)
        xout = self.bert(xcat, repr_layers=[self.num_esm_layers])
        xout  = xout["representations"][self.num_esm_layers]

        task_outputs = self.classifier.forward(xout)

        return task_outputs



# idx_to_char = ['pad','mask', 'cls', 'unk', 'L', 'A', 'G', 'V', 'S', 'I', 'E', 'R', 'D', 'T', 'K', 'P', 'F', 'N', 'Q', 'Y', 'H', 'M', 'W', 'C', 'X', 'U', 'O', 'Z', 'B']

# with open('mutations.csv','w') as f:
#     x1,conv_out1 = self.bert(x_in[0])
#     x2,conv_out2= self.bert(x_in[1])
#     task_outputs = self.classifier.forward(x1,x2)
#     reference = torch.sigmoid(task_outputs).item()
#     for j in range(437,508):
#         x1_seq = x_in[0]
#         min_score = 1
#         f.write(str(j)+',')
#         for i in range(4,29):
#             x1_new = x1_seq.clone()
#             x1_new[0][j] = i
#             x1,conv_out1 = self.bert(x1_new)
#             x2,conv_out2= self.bert(x_in[1])
#             task_outputs = self.classifier.forward(x1,x2)
#             out = torch.sigmoid(task_outputs).item()
#             diff = out - reference
#             f.write(str(diff)+',')
#             if out < min_score:
#                 min_score = out
#         print(j,min_score)
#         f.write('\n')



class OrderAgnosticClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden*2, hidden*2)
        self.nonlinear = nn.GELU()
        self.linear2 = nn.Linear(hidden*2, 1)

    def forward(self, x1,x2,x1_seq,x2_seq):
        x1 = x1[:,0,:].contiguous()
        x2 = x2[:,0,:].contiguous()

        # return 1/(1+self.l2(x1,x2))
        
        abs_diff = torch.abs(x1-x2)
        dot = x1*x2

        v_out = torch.cat((abs_diff,dot),1)
        v_out = self.linear1(v_out)
        v_out = self.nonlinear(v_out)
        v_out = self.dropout(v_out)
        out = self.linear2(v_out)

        return out.view(-1)

class OrderedClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden*2, int(hidden)*2)
        self.nonlinear = nn.ReLU()
        self.linear2 = nn.Linear(int(hidden)*2, 1)

    def forward(self, x1,x2,x1_seq=None,x2_seq=None):
        # x1 = x1[:,1:,:].mean(1)
        # x2 = x2[:,1:,:].mean(1)
        x1 = x1[:,0,:].contiguous()
        x2 = x2[:,0,:].contiguous()
        v_out = torch.cat((x1,x2),1)
        v_out = self.dropout(v_out)
        out = self.linear1(v_out)
        v_out = self.nonlinear(v_out)
        v_out = self.dropout(v_out)
        out = self.linear2(v_out)
        return out.view(-1)

# class OrderedClassifier(nn.Module):
#     def __init__(self, hidden, n_classes,dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(hidden*2, 1)

#     def forward(self, x1,x2,x1_seq=None,x2_seq=None):
#         x1 = x1[:,0,:].contiguous()
#         x2 = x2[:,0,:].contiguous()
#         v_out = torch.cat((x1,x2),1)
#         v_out = self.dropout(v_out)
#         out = self.linear1(v_out)
#         return out.view(-1)

class TransformerClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1,seq_len=512):
        super().__init__()
        n_layers = 2
        attn_heads = 8
        activation = 'gelu'

        self.transformer_blocks = nn.ModuleList([nn.TransformerEncoderLayer(hidden, attn_heads, dim_feedforward=hidden*4, dropout=dropout, activation=activation,eps=1e-12) for _ in range(n_layers)])

        self.position = PositionalEmbedding(d_model=hidden,max_len=seq_len)
        
        self.p1 = nn.Linear(hidden, hidden)
        self.p2 = nn.Linear(hidden, hidden)

        self.linear_out = nn.Linear(hidden, 1)


    def forward(self, x1,x2,x1_seq,x2_seq):
        x1 = x1[:,:,:].contiguous()
        x2 = x2[:,:,:].contiguous()

        x1 = self.p1(x1)
        x2 = self.p2(x2)

        # x1 = x1 + self.position(x1)
        # x2 = x2 + self.position(x2)

        x_concat = torch.cat((x1,x2),1)

        attn_mask = torch.zeros(x_concat.size(1),x_concat.size(1)).bool().cuda()

        # attn_mask[x1.size(1):,:]=True # disallow p2 to attend to anything
        # attn_mask[1:x1.size(1),1:x1.size(1)]=True # disallow p1 to attent to itself
        # attn_mask[0,x1.size(1):]=True # only let CLS attend to P1

        # attn_mask[:,0]=True # disallow all positions to attend to CLS
        # attn_mask[x1.size(1):,x1.size(1):]=True # disallow p2 to attent to itself

        key_mask = (torch.cat((x1_seq,x2_seq),1) == 0)

        x = x_concat.transpose(0,1)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, src_mask=attn_mask,src_key_padding_mask=key_mask)
        x = x.transpose(0,1)
        
        x = x[:,0,:].contiguous()
        out = self.linear_out(x)

        # x1_cls = x[:,0,:].contiguous()
        # x2_cls = x[:,x1.size(1),:].contiguous()
        # x1x2 = torch.cat((x1_cls,x2_cls),1)
        # out = self.linear_out(x1x2)

        return out.view(-1)


class RegressionClassifierModel(nn.Module):
    def __init__(self, bert: BERT, tasks, dropout,evo_bert=None):
        super().__init__()
        self.bert = bert
        self.evo_bert = evo_bert
        if evo_bert is not None:
            self.evo_bert = evo_bert
            self.classifier = RegressionClassifier(self.evo_bert.hidden,tasks)
        else:
            self.classifier = RegressionClassifier(self.bert.hidden,tasks)

    def forward(self, x_in, psi_mat=None):
        if self.bert is not None:
            x,conv_out = self.bert(x_in)

        if self.evo_bert is not None:
            x =  self.evo_bert(psi_mat,x_in)

        output = self.classifier.forward(x)
        return output

class RegressionClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden, n_classes)

        # self.dense = nn.Linear(hidden, hidden)
        # self.activation_fn = nn.GELU()
        # self.dropout = nn.Dropout(p=0.1)
        # self.layer_norm = nn.LayerNorm(hidden,eps=1e-12) 
        # self.out_proj = nn.Linear(hidden, n_classes)
    
    def forward(self, x):
        x = x[:,0,:].contiguous()
        x = self.linear1(x)
        return x

    # def forward(self, x):
    #     x = x[:,0,:].contiguous()
    #     x = self.dense(x)
    #     x = self.activation_fn(x)
    #     # x = self.dropout(x)
    #     x = self.layer_norm(x)
    #     x = self.out_proj(x)
    #     return x


class ContactClassifierModel(nn.Module):
    def __init__(self, bert: BERT, tasks, dropout,esm=False):
        super().__init__()
        self.bert = bert
        self.esm = esm
        if self.esm:
            self.num_esm_layers = len(bert.layers)
            hidden_size = bert.layers[-1].self_attn.out_proj.in_features
        else:
            hidden_size = self.bert.hidden

        self.classifier = PairwiseContactPredictionHead(hidden_size,tasks)
        

    def forward(self, x_in, psi_mat=None):
        if self.esm:
            x1 = self.bert(x_in, repr_layers=[self.num_esm_layers])
            x  = x1["representations"][self.num_esm_layers]
        else:
            x,conv_out = self.bert(x_in)
        
        task_outputs = self.classifier(x)

        return task_outputs

class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-100):
        super().__init__()
        self.k_linear = nn.Linear(hidden_size,hidden_size*4)
        self.q_linear = nn.Linear(hidden_size,hidden_size*4)

        self.k_linear_neg = nn.Linear(hidden_size,hidden_size*4)
        self.q_linear_neg = nn.Linear(hidden_size,hidden_size*4)

    def forward(self, inputs, sequence_lengths=None, targets=None):
        s_k = self.k_linear(inputs)
        s_q = self.q_linear(inputs)
        contacts = torch.matmul(s_k, s_q.transpose(1, 2))
        pos_pred = (contacts + contacts.transpose(1, 2)) / 2

        s_k_neg = self.k_linear_neg(inputs)
        s_q_neg = self.q_linear_neg(inputs)
        contacts_neg = torch.matmul(s_k_neg, s_q_neg.transpose(1, 2))
        neg_pred = (contacts_neg + contacts_neg.transpose(1, 2)) / 2

        outputs = torch.cat((pos_pred.unsqueeze(-1),neg_pred.unsqueeze(-1)),3)

        return outputs


 
class LanguageModel(nn.Module):
    def __init__(self, bert, n_classes, dropout,evo_bert=None):
        super().__init__()
        self.bert = bert
        self.evo_bert = evo_bert

        # if not esm: hidden_size = self.bert.hidden 
        hidden_size = 768
        self.num_esm_layers = 12
        n_classes = 35
        self.classifier = LMClassifier(hidden_size,n_classes)

    def forward(self, x_in, psi_mat=None):
        # x,conv_out = self.bert(x_in)
        # stop()
        xout = self.bert(x_in, repr_layers=[self.num_esm_layers])
        x  = xout["representations"][self.num_esm_layers]


        task_outputs = self.classifier.forward(x)

        return task_outputs

class LMClassifier(nn.Module):
    def __init__(self, hidden, n_classes,dropout=0.1):
        super().__init__()
        # self.classify = nn.Sequential(
        #     nn.Dropout(0.1, inplace=True),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.Linear(hidden, n_classes),
        #     )
        # self.dense = nn.Linear(hidden, hidden)
        # self.activation_fn = torch.tanh
        self.dropout = nn.Dropout(p=0.1)

        # hidden_size = bert.layers[-1].self_attn.out_proj.in_features
        self.out_proj = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x = self.classify(x)
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


nltbl = {'tanh' : nn.Tanh(),'relu': nn.ReLU(),'prelu': nn.PReLU()}


class conv_base_model(nn.Module):
    # base conv model-- pads, conv,nonlinear,maxpool --- this is one conv
    # layer
    def __init__(self, embedsize , kernelsize, hiddenunit, poolingsize, nonlinearity, convdropout):
        super().__init__()
        self.hiddenunit = hiddenunit
        self.poolingsize = poolingsize
        self.kernelsize = kernelsize
        self.embedsize = embedsize
        self.padding = int((self.kernelsize-1)/2)
        self.nonlinearity = nltbl[nonlinearity]
        self.convdropout = convdropout
        self.conv1d = nn.Conv1d(self.embedsize, self.hiddenunit, self.kernelsize)
        self.maxpool = nn.MaxPool1d(self.poolingsize)


    def forward(self, iput):
        # pad sequence  then duplicate padded with pooling 
        # iput2 = iput.unsqueeze(0)

        # padded_iput = torch.cat(([F.pad(iput2,(0,0,self.padding-j,self.kernelsize-self.padding-1+j+self.poolingsize)) for j in range(self.poolingsize)]),dim=0)
        padded_iput = torch.cat(([ torch.cat(([F.pad(ibatch.unsqueeze(0),(0,0,self.padding-j,self.kernelsize-self.padding-1+j+self.poolingsize)) for j in range(self.poolingsize)]),dim=0) for ibatch in iput]),dim=0)
        
        # padded_iput = padded_iput.view(-1,padded_iput.size(-2),padded_iput.size(-1))

        padded_iput = padded_iput.permute(0,2,1)
        # convolution 1D
        output = self.conv1d(padded_iput)
        output = self.nonlinearity(output)
        output = self.maxpool(output)
        output = F.dropout(output,p=self.convdropout,training=self.training)
        output = output.permute(0,2,1)
        return output



class ConvTokenClassifierModel(nn.Module):
    def __init__(self,n_classes, dropout):
        super().__init__()

        self.lookuptable = nn.Embedding(24,24)

        
        self.conv_base = conv_base_model( 24, 5, 1024, 2, 'relu', 0.1)
        conv_layers = []
        self.nlayers = 1
        for i in range(1,self.nlayers):
            conv_layers.append(conv_base_model(1024 , 5, 1024, 2, 'relu', 0.1))
        self.conv_layers = nn.Sequential(*conv_layers)

        # linear layers for different tasks 1kernel size is basically linear layer 

        self.classifier = nn.Linear(1024,n_classes)


    def forward(self, seq,evo):
        # seq_len = sequences.size()[0]
        emb = self.lookuptable(seq.long())
        emb = F.dropout(emb,p=0.1,training=self.training)
        # stop()
        # pad,conv,stitch
        output = self.conv_base(emb)
        if self.nlayers > 1: output = self.conv_layers(output)
        # stitch together sequence

        # stop()
        # output = output.contiguous().view(seq.size(0),-1,output.size(-1))[:,0:seq.size(1),:]
        # L,P,H = output.size()
        # output = (output.permute(2,1,0).contiguous().view(H,P*L,1))
        # output = output.permute(2,0,1)

        # predict and make output same size as input
        output=self.classifier(output)
        
        return output

