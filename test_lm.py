import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm, random, string, os, time, math
from pdb import set_trace as stop 
from utils.metrics import evaluate
import logging
from utils.logger import setup_logger
import scipy.stats as ss

class LMTester:
    """LMTester class, used for LM training, test, and validation
    """
    def __init__(self,args,model_and_data):
        """Constructor
        """
        self.device = args.device
        self.task = args.task
        self.model = model_and_data['model']
        self.vocab = args.vocab

        data = model_and_data['data']
        self.train_data = data['train']
        self.valid_data = data['valid'] 
        self.test_data = data['test']
        
        self.train_logger = setup_logger(name='train', log_file=args.model_name+'/train.log')
        self.valid_logger = setup_logger(name='valid', log_file=args.model_name+'/valid.log')
        self.test_logger = setup_logger(name='test', log_file=args.model_name+'/test.log')
        self.model_name = args.model_name 
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, evalu=False,max_batches=-1):
        """Trains the model
        :param epoch: Number of epochs to train th emodel for
        :param max_batches: Number of batches to train the model for
        """
        return self.iteration(epoch,self.train_data,self.train_logger,train=True,split_name='Train')

    def test(self, epoch, evalu=False,max_batches=-1):
        """Tests the model
        :param epoch: Number of epochs to test the model for
        :param max_batches: Number of batches to train the model for
        """
        return self.iteration(epoch,self.test_data,self.test_logger,train=False,split_name='Test')

    def valid(self, epoch, evalu=False,max_batches=-1):
        """Validates the model
        :param epoch: Number of epochs to validate the model for
        :param max_batches: Number of batches to validate the model for
        """
        return self.iteration(epoch,self.valid_data,self.valid_logger,train=False,split_name='Valid')

    def iteration(self, epoch,data_loader,logger,train=True,split_name=''):
         """Runs each iteration of the model 
        :param epoch: Number of epochs to train the model for
        :param data_loader: Data that has been loaded by the model
        """
        self.model.eval()

        data_iter = tqdm.tqdm(enumerate(data_loader),desc="%s" % (split_name),total=len(data_loader),bar_format="{l_bar}{r_bar}")
        grammars = []
        preferences = []
        prediction = -1
        for i, data in data_iter:
            if i == 10000:
                break
            wt_seq = data['wt_seq']
            mut_seq = data['mut_seq']
            mut_pos = data['mut_pos']
            preference = data['preference']

            prediction = self.model.forward(wt_seq[0:1])
            prediction = prediction[0].detach().cpu()
            
            for idx,mut_positions in enumerate(mut_pos):     
                if True:
                    raw_probs = []
                    for pos in mut_positions:
                        mut_aa = mut_seq[idx][pos+1].item()
                        logit = prediction[pos+1][mut_aa]
                        prob = torch.sigmoid(logit)
                        raw_probs.append(prob.item())
            
                    grammar = sum(np.log10(raw_probs))
                    grammars.append(grammar)
                    preferences.append(preference[idx].item())

        print(len(grammars))
        spearman = ss.spearmanr(preferences, grammars)
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        plt.scatter(preferences,grammars,s = 1,c='blue')
        plt.savefig('spearman.png')
        print(spearman)
        exit()
        
        metrics = evaluate(self.task,[],[],total_loss,total_correct,total_preds)
        
        return metrics
        


