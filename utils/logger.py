
import numpy
import scipy.sparse as sp
import scipy
import logging
from collections import OrderedDict
import sys
import torch
import math
from pdb import set_trace as stop
import os
# import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch.nn.functional as F
import json
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
	
    handler = logging.FileHandler(log_file, mode='w')   
	# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')    
    # handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class Logger(object):
    def __init__(self, log_dir,save_best):
        self.log_dir = log_dir
        self.best_valid_acc = -math.inf
        self.best_test_acc = -math.inf
        self.best_epoch_flag = False
        self.save_best = save_best

        self.train_dict = {}
        self.valid_dict = {}
        self.test_dict = {}

    def print_metrics(self,metrics,verbose=False):
        for key,value in metrics.items():
            # print(key+': ',"%.3f" % value)
            if isinstance(value,dict):
                metric_vals = list(value.values())
                metric_mean = np.array(metric_vals).mean()
                print("{:7}{:.3f}".format(key+':',metric_mean))
                ordered_dict = OrderedDict(sorted(value.items()))
                if verbose:
                    for k2,v2 in ordered_dict.items(): 
                        print("{:7}{:.3f}".format(k2+',',v2))
                    
            else:
                print("{:7}{:.3f}".format(key+':',value))


    def log_train(self,metrics,epoch):
        self.print_metrics(metrics,verbose=False)
        self.train_dict[epoch] = metrics
        with open(os.path.join(self.log_dir,'train_log.json'),'w') as fp:
            json.dump(self.train_dict, fp)

                
    def log_valid(self,metrics,epoch,trainer):
        self.print_metrics(metrics,verbose=False)
        self.valid_dict[epoch] = metrics
        with open(os.path.join(self.log_dir,'valid_log.json'),'w') as fp:
            json.dump(self.valid_dict, fp)
        
        if metrics['acc'] > self.best_valid_acc:
            self.best_valid_acc = metrics['acc']
            self.best_epoch_flag = True
            if self.save_best:
                output_path = self.log_dir + "/best_model.pt"
                torch.save(trainer.model.cpu().state_dict(), output_path)
                trainer.model.to(trainer.device)
                print("EP: {} Model Saved at: {}\n".format(epoch, output_path))
        else:
            self.best_epoch_flag = False
            
    
    def log_test(self,metrics,epoch,printvals=True):
        if printvals:
            self.print_metrics(metrics,verbose=False)
        self.test_dict[epoch] = metrics
        with open(os.path.join(self.log_dir,'test_log.json'),'w') as fp:
            json.dump(self.test_dict, fp)
        if self.best_epoch_flag:
            self.best_metrics = metrics
        print('\n****** BEST RESULTS ********')
        self.print_metrics(self.best_metrics)
        print('****************************')


class MultiLogger(object):
    def __init__(self, log_dir,save_best,data):
        self.log_dir = log_dir
        self.save_best = save_best
        
        self.best_valid_loss = float("inf")
        self.log_dict = {}
        for dataset in data:
            task = dataset['task']
            self.log_dict[task] = {}
            self.log_dict[task]['best_valid_acc'] = 0
            self.log_dict[task]['best_test_acc'] = 0
            self.log_dict[task]['best_epoch_flag'] = False

            self.log_dict[task]['train_dict'] = {}
            self.log_dict[task]['valid_dict'] = {}
            self.log_dict[task]['test_dict'] = {}

    def print_metrics(self,task,metrics):
        for key,value in metrics.items():
            print("{}: {:7}{:.3f}".format(task,key+':',value))
        print()

    def log_train(self,metrics,epoch):
        
        for task,task_metrics in metrics.items(): 
            self.print_metrics(task,task_metrics)
            self.log_dict[task]['train_dict'][epoch] = task_metrics

            with open(os.path.join(self.log_dir,'train_log_'+task+'.json'),'w') as fp:
                json.dump(self.log_dict[task]['train_dict'], fp)

                
    def log_valid(self,metrics,epoch,trainer):
        epoch_loss = 0
        for task,task_metrics in metrics.items(): 
            self.print_metrics(task,task_metrics)
            self.log_dict[task]['valid_dict'][epoch] = task_metrics

            with open(os.path.join(self.log_dir,'valid_log.json'),'w') as fp:
                json.dump(self.log_dict[task]['valid_dict'], fp)
        
            epoch_loss += task_metrics['loss']
            if task_metrics['acc'] > self.log_dict[task]['best_valid_acc']:
                self.best_valid_acc = self.log_dict[task]['best_valid_acc']
                self.log_dict[task]['best_epoch_flag'] = True
            else:
                self.log_dict[task]['best_epoch_flag'] = False
        
        if epoch_loss < self.best_valid_loss:
            self.best_valid_loss = epoch_loss
            print('best loss: ',epoch_loss)
            if self.save_best:
                output_path = self.log_dir + "/best_model.pt"
                torch.save(trainer.model_and_data[0]['model'].cpu().state_dict(), output_path)
                trainer.model_and_data[0]['model'].to(trainer.device)
                print("EP:%d Model Saved on:" % epoch, output_path)
                print()
        
    
    def log_test(self,metrics,epoch):
        for task,task_metrics in metrics.items(): 
            self.print_metrics(task,task_metrics)
            self.log_dict[task]['test_dict'][epoch] = task_metrics
            with open(os.path.join(self.log_dir,'test_log.json'),'w') as fp:
                json.dump(self.log_dict[task]['test_dict'], fp)

            if self.log_dict[task]['best_epoch_flag']:
                self.log_dict[task]['best_test_acc'] = task_metrics['acc']

        print('**********************')
        for task,task_metrics in metrics.items(): 
            print("{} best test acc: {:.3f}".format(task, self.log_dict[task]['best_test_acc']))
        print('**********************')

