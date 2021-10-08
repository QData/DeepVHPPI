import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .bert import BERT
from pdb import set_trace as stop
from .models import *
from .utils import weights_init
from collections import OrderedDict
import fb_esm as esm



def load_bert(file_name,model):
    print('Loading BERT Params from: '+file_name)
    module_state_dict = torch.load(file_name)
    module_state_dict = OrderedDict([(k.replace('module.',''), v) for k,v in module_state_dict.items()])
    bert_state_dict = [((k.replace('bert.',''), v) if 'bert' in k else None) for k, v in module_state_dict.items()]
    bert_state_dict = [i for i in bert_state_dict if i]
    bert_state_dict = OrderedDict(bert_state_dict)
    # bert_state_dict['embedding.position.pe'] = model.bert.embedding.position.pe
    model.bert.load_state_dict(bert_state_dict)
    return model

def load_model(file_name,model):
    print('Loading Model Params from: '+file_name)
    module_state_dict = torch.load(file_name)
    module_state_dict = OrderedDict([(k.replace('module.',''), v) for k,v in module_state_dict.items()])
    model.bert.contact_head = None
    model.load_state_dict(module_state_dict)
    return model

def get_bert(args,vocab_size):
    bert = BERT(vocab_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads,max_len=args.seq_len2,dropout=args.dropout,emb_type=args.emb_type,activation=args.activation)

    return bert

def get_classifier(bert,task,args,vocab_size,num_classes,device):
    if task in ['secondary','transmembrane','4prot','malaria']:
        model = TokenClassifierModel(bert,num_classes,args.dropout,esm=args.esm)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif task in ['homology','solubility','localization']:
        model = SequenceClassifierModel(bert,num_classes,args.dropout,esm=args.esm)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    elif task in ['fluorescence','stability']:
        model = RegressionClassifierModel(bert,num_classes,args.dropout)
        criterion = nn.MSELoss()
    elif task == 'contact':
        model = ContactClassifierModel(bert,num_classes,args.dropout,esm=args.esm)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    elif task in ['biogrid','ppi']:
        model = PairwiseClassifierModel(bert,num_classes,args.dropout,seq_len=args.seq_len,esm=args.esm)
        
        # criterion = nn.BCEWithLogitsLoss(weight=torch.Tensor([5]).cuda())
        criterion = nn.BCEWithLogitsLoss()
    elif task in ['lm','test_lm']:
        model = LanguageModel(bert,num_classes,args.dropout)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        print('No model for Task {}. Exiting.'.format(task))
        exit(0)

    if args.task == 'multi':
        print(model.classifier)
    else:
        print(model)

    if args.esm:
        print('RESETING CLASSIFIER WEIGHTS')
        model.classifier.apply(weights_init)
        

    if args.reset_weights or (not args.esm):
        print('RESETING ALL WEIGHTS')
        model.apply(weights_init)

    if args.saved_bert != '':
        model = load_bert(args.saved_bert,model)

    if args.saved_model != '':
        model = load_model(args.saved_model,model)

    if args.freeze_bert:
        print('Freezing BERT Params')
        for param in model.bert.parameters():
            param.requires_grad = False


    model = model.to(device)

    if args.with_cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=args.cuda_devices)

    
    return model,criterion

def initialize_model_single(args,task,vocab_size,data,device,bert=None):
    print("Building BERT model")
    if bert == None:
        bert = get_bert(args,vocab_size)

    model,criterion = get_classifier(bert,task,args,vocab_size,data['num_classes'],device)

    model_and_data = {}
    model_and_data['model'] = model
    model_and_data['criterion'] = criterion
    model_and_data['data'] = data

    return model_and_data

def initialize_model_multi(args,vocab_size,data,device,bert=None):
    print("Building MultiTask BERT model")
    
    if bert == None:
        bert = get_bert(args,vocab_size)
    
    model_and_data_list = []
    for dataset in data:
        print(dataset['task'])
        model,criterion = get_classifier(bert,dataset['task'],args,vocab_size,dataset['num_classes'],device)
        model_and_data = {}
        model_and_data['model'] = model
        model_and_data['criterion'] = criterion
        model_and_data['data'] = dataset
        model_and_data_list.append(model_and_data)
    
    return model_and_data_list

def initialize_model(args,vocab_size,data,device,bert):
    if args.task == 'multi':
        return initialize_model_multi(args,vocab_size,data,device,bert)
    else:
        return initialize_model_single(args,args.task,vocab_size,data,device,bert)