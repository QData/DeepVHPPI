import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import tqdm, random, string, os, time, math
from pdb import set_trace as stop 
from collections import OrderedDict
from utils.metrics import evaluate
from utils.optim_schedule import WarmupLinearSchedule
import scipy
from matplotlib import pyplot as plt


class BERTTrainer:
    """BERT Trainer class, used to train the model
    """
    def __init__(self,args,model_and_data):
        """Constructor
        """
        lr = args.lr
        self.device = args.device
        self.task = args.task
        self.model = model_and_data['model']
        self.criterion = model_and_data['criterion']
        self.grad_ac_steps=args.grad_ac_steps

        self.train_data = model_and_data['data']['train']
        self.valid_data = model_and_data['data']['valid'] 
        self.test_data = model_and_data['data']['test']

        if args.optimizer == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=args.adam_weight_decay)
        else:
            self.optim = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=0.9)
        self.scheduler_warmup = WarmupLinearSchedule( self.optim, args.warmup_steps, 100000000000)

        self.model_name = args.model_name
        self.update_steps = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, max_batches=-1, evalu=False):
        """Trains the model
        :param epoch: Number of epochs to train the model for
        :param max_batches: Number of batches to train the model for
        """
        return self.iteration(epoch, max_batches, self.train_data,train=True,split_name='Train', evalu=evalu)

    def test(self, epoch, max_batches=-1, evalu=False):
        """Tests the model
        :param epoch: Number of epochs to test the model for
        :param max_batches: Number of batches to test the model for
        """
        return self.iteration(epoch, max_batches, self.test_data,train=False,split_name='Test', evalu=evalu)

    def valid(self, epoch, max_batches=-1, evalu=False):
        """Validates the model
        :param epoch: Number of epochs to validate the model for
        :param max_batches: Number of batches to validate the model for
        """
        return self.iteration(epoch, max_batches, self.valid_data,train=False,split_name='Valid', evalu=evalu)

    def iteration(self, epoch, max_batches, data_loader, train=True,split_name='',evalu=False):
        """Runs each iteration of the model 
        :param epoch: Number of epochs to train the model for
        :param data_loader: Data that has been loaded by the model
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_preds = 0
        total_correct = 0
        all_preds = []
        all_targets = []
        all_targets_ka = []
        all_seq_lens = []
        all_seq1_ids = []
        all_seq2_ids = []

        self.optim.zero_grad()
        data_iter = tqdm.tqdm(enumerate(data_loader),desc="%s" % (split_name),total=len(data_loader),bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            if i == max_batches: break

            task_inputs = (data["bert_input"][0].to(self.device),data["bert_input"][1].to(self.device),data["bert_input"][2].to(self.device))
            seq1_ids = data["input_id"][0]
            seq2_ids = data["input_id"][1]
            all_seq1_ids += seq1_ids
            all_seq2_ids += seq2_ids
            target = data['bert_label'].to(self.device)

            target_ka = None
            if data['bert_label_continuous'] is not None:
                target_ka = data['bert_label_continuous']

            sequence_lengths = data['line_len']
            
            if train:
                prediction = self.model.forward(task_inputs, target)
            else:
                with torch.no_grad():
                    prediction = self.model.forward(task_inputs, target)
                

            loss = self.criterion(prediction,target.float())
            total_loss += loss.item()
            total_preds += len(target)
            
            if train: 
                loss.backward()
                if ((i+1)%self.grad_ac_steps == 0):
                    self.optim.step()
                    self.optim.zero_grad()
                    self.scheduler_warmup.step()
                    self.update_steps+=1   

            prediction = torch.sigmoid(prediction)
            prediction = prediction.view(-1).detach().cpu()
            target_out = target.view(-1).detach().cpu()
            all_preds += prediction.tolist()
            all_targets += target_out.tolist()
 
        metrics = evaluate(self.task,
                            all_preds,
                            all_targets,
                            total_loss,
                            total_correct,
                            total_preds,
                            all_seq_lens=all_seq_lens,
                            train=train,
                            split_name=split_name,
                            model_name=self.model_name,
                            seq1_ids=all_seq1_ids,
                            seq2_ids=all_seq2_ids)


        return metrics
        


