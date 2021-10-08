import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm, random, string, os, time, math
from pdb import set_trace as stop 
from collections import OrderedDict
from utils.metrics import evaluate
from utils.optim_schedule import WarmupLinearSchedule
 
class BERTMultitaskTrainer:
    """BERT Multitaks Trainer class, used to train the model
    """
    def __init__(self,args,model_and_data):
        """Constructor
        """
        task = args.task
        self.device = args.device
        self.grad_ac_steps=args.grad_ac_steps

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        for model_dict in model_and_data:
            if args.optimizer == 'adam':
                optim = torch.optim.Adam(model_dict['model'].parameters(),lr=args.lr,weight_decay=args.adam_weight_decay)
            else:
                optim = torch.optim.SGD(model_dict['model'].parameters(),lr=args.lr,momentum=0.9)

            model_dict['optim'] = optim

            model_dict['scheduler_warmup'] = WarmupLinearSchedule(optim, args.warmup_steps, 1000000)

        self.model_and_data = model_and_data

        self.model_name = args.model_name
        self.update_steps = 0

        # print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, max_batches=-1, evalu=False):
        """Trains the model
        :param epoch: Number of epochs to train the model for
        :param max_batches: Number of batches to train the model for
        """
        return self.iteration(epoch, self.train_data,train=True,split_name='train', evalu=evalu)

    def test(self, epoch, max_batches=-1, evalu=False):
        """Tests the model
        :param epoch: Number of epochs to test the model for
        :param max_batches: Number of batches to test the model for
        """
        return self.iteration(epoch, self.test_data,train=False,split_name='test', evalu=evalu)

    def valid(self, epoch, max_batches=-1, evalu=False):
        """Validates the model
        :param epoch: Number of epochs to validate the model for
        :param max_batches: Number of batches to validate the model for
        """
        return self.iteration(epoch, self.valid_data,train=False,split_name='valid', evalu=evalu)

    def iteration(self, epoch, data_loader, train=True,split_name='',evalu=False):
        """Runs each iteration of the model 
        :param epoch: Number of epochs to train the model for
        :param data_loader: Data that has been loaded by the model
        """
        results_dict = {}
        task_idxs = []
        task_lens = []
        for task_idx,md in enumerate(self.model_and_data):
            task = md['data']['task']
            if train:
                md['model'].train()
            else:
                md['model'].eval()
            
            task_idxs += [task_idx]*len(md['data'][split_name])
            task_lens += [len(md['data'][split_name])]
            results_dict[task] = {}
            results_dict[task]['total_loss'] = 0
            results_dict[task]['total_preds'] = 0
            results_dict[task]['total_correct'] = 0
            results_dict[task]['all_preds'] = []
            results_dict[task]['all_targets'] = []
            results_dict[task]['all_seq_lens'] = []
            
        total_batches = len(task_idxs)
        task_lens = np.power(np.array(task_lens),0)
        task_ratios = task_lens/task_lens.sum()

        data_iter = tqdm.tqdm(task_idxs,desc="%s" % (split_name),total=total_batches,bar_format="{l_bar}{r_bar}")
        for batch_idx,task_idx in enumerate(data_iter):
            if train:
                # task_idx = batch_idx % len(self.model_and_data) # sample uniformly
                np.random.choice(np.arange(0,len(task_ratios)), p=task_ratios) # sample randomly
            md = self.model_and_data[task_idx]
            batch = next(iter(md['data'][split_name]))
            task=md['data']['task']

            task_inputs = batch["bert_input"].to(self.device)
            evo = batch['bert_evo'].float().to(self.device)
            target = batch['bert_label'].to(self.device)
            sequence_lengths = batch['line_len'].to(self.device)
            
            # stop()
            if not train:
                with torch.no_grad():
                    prediction = md['model'].forward(task_inputs, target)
            else:
                prediction = md['model'].forward(task_inputs, target)


            if task in ['fluorescence','stability','covid']:
                loss = md['criterion'](prediction,target.float())
                results_dict[task]['total_loss'] += loss.item()
                results_dict[task]['total_preds'] += len(target)
            else:
                loss = md['criterion'](prediction.view(-1,prediction.size(-1)),target.view(-1))
                sum_loss_fun = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')
                batch_sum_loss = sum_loss_fun(prediction.view(-1,prediction.size(-1)),target.view(-1))
                results_dict[task]['total_loss'] += batch_sum_loss.item()
                nonzero_targets = len(target[target != -1]) 
                results_dict[task]['total_preds'] += nonzero_targets

            if train: 
                loss.backward()
                # if ((i+1)%self.grad_ac_steps == 0):
                md['optim'].step()
                md['optim'].zero_grad()
                md['scheduler_warmup'].step(self.update_steps)
                self.update_steps+=1 
            
            if task in ['secondary','homology','4prot','solubility','localization','transmembrane']:
                _,pred_max = prediction.view(-1,prediction.size(-1)).detach().cpu().max(1)
                target_out = target.view(-1).detach().cpu()[target.view(-1)>-1]
                pred_max = pred_max[target.detach().cpu().view(-1)>-1] 
                results_dict[task]['all_preds'] += pred_max.tolist()
                results_dict[task]['all_targets']+= target_out.tolist()
            elif task in ['fluorescence','stability','covid']:
                prediction = prediction.view(-1).detach().cpu()
                target_out = target.view(-1).detach().cpu()
                results_dict[task]['all_preds'] += prediction.tolist()
                results_dict[task]['all_targets'] += target_out.tolist()
            elif task == 'contact':
                if not train:
                    results_dict[task]['all_seq_lens'] += sequence_lengths.tolist()
                    results_dict[task]['all_preds'].append(prediction.detach().cpu().numpy())
                    results_dict[task]['all_targets'].append(target.detach().cpu().numpy())
        
        task_metrics = {}
        for md in self.model_and_data:
            task = md['data']['task']
        
            metrics = evaluate(task,
                                results_dict[task]['all_preds'],
                                results_dict[task]['all_targets'],
                                results_dict[task]['total_loss'],
                                results_dict[task]['total_correct'],
                                results_dict[task]['total_preds'],
                                all_seq_lens=results_dict[task]['all_seq_lens'],
                                train=train)
            
            task_metrics[task] = metrics

        return task_metrics
    