import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm, random, string, os, time, math
from pdb import set_trace as stop 
from collections import OrderedDict
from utils.metrics import evaluate
import logging
from utils.logger import setup_logger
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from utils.optim_schedule import WarmupLinearSchedule

class BERTPreTrainer:
     """ProteinBERT Trainer class, used to train the model
    """
    def __init__(self,
                 model_and_data,
                 optimizer,
                 task,
                 lr: float = 2e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_steps=1,
                 device='cpu',
                 log_freq: int = 100, 
                 model_name='',
                 grad_ac_steps=1,
                 ):

        self.device = device
        self.task = task
        self.model = model_and_data['model']
        self.criterion = model_and_data['criterion']
        self.log_freq = log_freq
        self.grad_ac_steps=grad_ac_steps

        self.train_data = model_and_data['data']['train']
        self.valid_data = model_and_data['data']['valid'] 
        self.test_data = model_and_data['data']['test']
    
        
        self.train_logger = setup_logger(name='train', log_file=model_name+'/train.log')
        self.valid_logger = setup_logger(name='valid', log_file=model_name+'/valid.log')
        self.test_logger = setup_logger(name='test', log_file=model_name+'/test.log')

        if optimizer == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        else:
            self.optim = torch.optim.SGD(self.model.parameters(),lr=lr,momentum=0.9)
            
        self.scheduler_warmup = WarmupLinearSchedule( self.optim, warmup_steps, 100000)

        self.model_name = model_name 
        self.update_steps = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, evalu=False,max_batches=-1):
        """Trains the model
        :param epoch: Number of epochs to train th emodel for
        :param max_batches: Number of batches to train the model for
        """
        return self.iteration(epoch,self.train_data,self.train_logger,train=True,split_name='Train',max_batches=max_batches)

    def test(self, epoch, evalu=False,max_batches=-1):
        """Tests the model
        :param epoch: Number of epochs to test the model for
        :param max_batches: Number of batches to test the model for
        """
        return self.iteration(epoch,self.test_data,self.test_logger,train=False,split_name='Test')

    def valid(self, epoch, evalu=False,max_batches=-1):
        """Validates the model
        :param epoch: Number of epochs to validate the model for
        :param max_batches: Number of batches to validate the model for
        """
        return self.iteration(epoch,self.valid_data,self.valid_logger,train=False,split_name='Valid')

    def iteration(self, epoch,data_loader,logger,train=True,split_name='',max_batches=-1):
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
        
        batch_correct = 0
        batch_preds = 0
        batch_sum_loss = 0
        
        data_iter = tqdm.tqdm(enumerate(data_loader),desc="%s" % (split_name),total=len(data_loader),bar_format="{l_bar}{r_bar}")
        self.optim.zero_grad()
        for i, data in data_iter:
            task_inputs = data["bert_input"].to(self.device)
            target = data['bert_label'].to(self.device)
            evo = data['bert_evo'].float().to(self.device)
            sequence_lengths = data['line_len'].to(self.device)
            ace2_interaction = data['ace2_interaction']
            prediction = self.model.forward(task_inputs, evo)
            loss = self.criterion(prediction.view(-1,prediction.size(-1)),target.view(-1))

            if train: 
                loss.backward()
                if ((i+1)%self.grad_ac_steps == 0):
                    self.optim.step()
                    self.optim.zero_grad()
                    self.scheduler_warmup.step(self.update_steps)
                    self.update_steps+=1

            prediction = prediction.detach().cpu()
            target = target.detach().cpu()
            sum_loss_fun = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')
            sum_loss = sum_loss_fun(prediction.view(-1,prediction.size(-1)),target.view(-1))
    
            total_loss += sum_loss.item()
            batch_sum_loss += sum_loss.item()

            _,pred_max = prediction.view(-1,prediction.size(-1)).max(1)
            target_out = target.view(-1)[target.view(-1)>-1]
            pred_max = pred_max[target.view(-1)>-1] 

            correct = (pred_max == target_out).sum().item()
            total_correct += correct
            batch_correct += correct

            nonzero_targets = len(target[target != -1])
            total_preds += nonzero_targets
            batch_preds += nonzero_targets
            
            del prediction

            if ((i+1)%self.grad_ac_steps == 0):     
                acc = batch_correct/batch_preds
                batch_ppl = torch.exp(torch.Tensor([batch_sum_loss/batch_preds])).item()
                data_iter.write("Ep{} {}: ({}/{}): acc={:.2f}, ppl={:.2f}, lr={:.2e}  {}".format(epoch,split_name,i,len(data_loader),acc,batch_ppl,self.optim.param_groups[0]['lr'],self.model_name.split('/')[-1]))
                
                logger.info("{},acc={:.2f},ece={:.2f}".format(i,acc,batch_ppl))

                batch_correct = 0
                batch_preds = 0
                batch_sum_loss = 0

            if i == max_batches:
                break

        
        metrics = evaluate(self.task,
                           [],
                           [],
                           total_loss,
                           total_correct,
                           total_preds)


        return metrics
        


