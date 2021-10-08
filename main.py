import argparse
import torch
import os
from model import initialize_model
from dataset import load_data
from data import WordVocab
from pdb import set_trace as stop
from config_args import get_args
from train_lm import BERTPreTrainer
from test_lm import LMTester
from train_multitask import BERTMultitaskTrainer
import utils
import logging
import fb_esm as esm

args = get_args(argparse.ArgumentParser())

cuda_condition = torch.cuda.is_available() and args.with_cuda
args.device = torch.device("cuda:0" if cuda_condition else "cpu")

print("Loading Vocab", args.vocab_path)
vocab = WordVocab.load_vocab(args.vocab_path)
args.vocab = vocab

vocab_size = len(vocab)
print("Vocab Size: ", vocab_size)
print(args.model_name)

if args.esm:
	esm_bert, esm_alphabet = esm.pretrained.esm1_t12_85M_UR50S()
	esm_batch_converter = esm_alphabet.get_batch_converter()
else:
	esm_bert,esm_alphabet,esm_batch_converter = None,None,None


data = load_data(args,vocab,esm_alphabet,esm_batch_converter)

model_and_data = initialize_model(args,vocab_size,data,args.device,esm_bert)

print("Creating Trainer")
if args.pretrain:
	Logger = utils.Logger(args.model_name,args.save_best)
	runner = BERTPreTrainer(model_and_data,
					  optimizer=args.optimizer,
					  task=args.task,
					  lr=args.lr,
					  betas=(args.adam_beta1, args.adam_beta2), 
					  weight_decay=args.adam_weight_decay,
					  warmup_steps=args.warmup_steps,
					  device=args.device,
					  log_freq=args.log_freq,
					  model_name=args.model_name,
					  grad_ac_steps=args.grad_ac_steps,
					 )
elif args.test_lm:
	Logger = utils.Logger(args.model_name,args.save_best)
	runner = LMTester(args,model_and_data)

elif args.task == 'multi':
	Logger = utils.MultiLogger(args.model_name,args.save_best,data)
	runner = BERTMultitaskTrainer(args,model_and_data)
else:
	if args.task in ['biogrid','ppi']:
		from train_pair import BERTTrainer
		Logger = utils.Logger(args.model_name,args.save_best)
		runner = BERTTrainer(args,model_and_data)
	else:
		from train_single import BERTTrainer
		Logger = utils.Logger(args.model_name,args.save_best)
		runner = BERTTrainer(args,model_and_data)


print("Training Start")
print(args.model_name)
for epoch in range(args.epochs):
	print('\n================ Epoch '+str(epoch+1)+' ====================')
	print(args.model_name)
	if args.task != 'multi' and hasattr(runner, 'optim'):
		for param_group in runner.optim.param_groups:
			print(param_group['lr'])
		print() 

	if args.train_dataset is not None:
		print('\n==> Train')
		train_metrics = runner.train(epoch,max_batches=args.max_batches)
		Logger.log_train(train_metrics,epoch)
		if args.pretrain or 'ALL' in args.model_name:
			print('Saving')
			output_path = args.model_name + "/best_model.pt"
			torch.save(runner.model.cpu().state_dict(), output_path)
			runner.model.to(runner.device)

	if args.valid_dataset is not None:
		print('\n==> Valid')
		valid_metrics = runner.valid(epoch,max_batches=args.max_batches)
		Logger.log_valid(valid_metrics,epoch,runner)

		print('\n==> Test')
		if args.test_dataset is not None: 
			test_metrics = runner.test(epoch,max_batches=args.max_batches)
			Logger.log_test(test_metrics,epoch)
		else:
			Logger.log_test(valid_metrics,epoch,printvals=False)
	else:
		Logger.log_valid(train_metrics,epoch,runner)
		Logger.log_test(train_metrics,epoch)
