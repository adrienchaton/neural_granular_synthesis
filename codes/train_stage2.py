#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:17:53 2021

@author: adrienbitton
"""



from argparse import ArgumentParser
import os
import shutil 
import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from models import waveform_model,latent_model
from utils_stage1 import make_audio_dataloaders,export_latents
from utils_stage2 import make_latent_dataloaders,export_embedding_to_audio_reconstructions,export_random_samples,export_embeddings,plot_embeddings
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('Agg') # for the server
from matplotlib import pyplot as plt
import numpy as np
import json
import time



pl.seed_everything(1234)
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
curr_dir = os.getcwd()



# ------------
# hyper-parameters and trainer
# ------------

parser = ArgumentParser()
parser.add_argument('--mname', default="test_latent_model", type=str)
parser.add_argument('--waveform_mname', default="test_waveform_model", type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--max_steps', default=100000, type=int)
parser.add_argument('--tar_beta', default=1., type=float)
parser.add_argument('--beta_steps', default=500, type=int)
parser.add_argument('--conditional', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--gpus', default=1, type=int)
parser.add_argument('--precision', default=32, type=int)
parser.add_argument('--profiler', action='store_true')
parser.add_argument('--out_dir', default="outputs", type=str)
parser.add_argument('--l_config', default="./configs/another_config.json", type=str)
args = parser.parse_args()

args.mname = args.waveform_mname+"__"+args.mname

with open(args.l_config) as json_file:
    l_config = json.load(json_file)

default_root_dir = os.path.join(curr_dir, args.out_dir, args.mname)
print("writing outputs into default_root_dir",default_root_dir)

# lighting is writting output files in default_root_dir/lightning_logs/version_0/
tmp_dir = os.path.join(default_root_dir,"lightning_logs","version_0")


###############################################################################
## STAGE 1: loading configuration and parameters of waveform VAE + creating dataset of latent projections
###############################################################################

print("\n*** loading of pretrained waveform VAE from",os.path.join(curr_dir, args.out_dir, args.waveform_mname))

w_args = np.load(os.path.join(curr_dir, args.out_dir, args.waveform_mname, "argparse.npy"),allow_pickle=True).item()

with open(w_args["w_config"]) as json_file:
    w_config = json.load(json_file)

print("\n*** loading audio data")

train_dataloader,test_dataloader,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(w_args["data_dir"],
                            w_args["classes"],w_config["sr"],w_config["silent_reject"],w_config["amplitude_norm"],args.batch_size,tar_l=w_args["tar_l"],l_grain=w_config["l_grain"],
                            high_pass_freq=50.,num_workers=args.num_workers)

print("\n*** restoring waveform model checkpoint")

ckpt_file = sorted(glob.glob(os.path.join(curr_dir, args.out_dir, args.waveform_mname, "checkpoints", "*.ckpt")))[-1]
yaml_file = os.path.join(curr_dir, args.out_dir, args.waveform_mname, "hparams.yaml")
w_model = waveform_model.load_from_checkpoint(checkpoint_path=ckpt_file,hparams_file=yaml_file,map_location='cpu')

w_model.to(device)
w_model.eval()
w_model.freeze()


# ------------
# data
# ------------

print("\n*** exporting latent projections")

train_latents,train_labels,test_latents,test_labels = export_latents(w_model,train_dataloader,test_dataloader)

print("\n*** preparing latent dataloaders")

train_latentloader,test_latentloader = make_latent_dataloaders(train_latents,train_labels,test_latents,test_labels,args.batch_size,num_workers=args.num_workers)


###############################################################################
## STAGE 2: training latent VAE
###############################################################################

print("\n*** STAGE 2: training latent VAE")

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(max_steps=args.max_steps,check_val_every_n_epoch=1,
                     gpus=args.gpus,precision=args.precision,benchmark=True,
                     default_root_dir=default_root_dir,profiler=args.profiler,
                     progress_bar_refresh_rate=50,callbacks=[lr_monitor])


# ------------
# model
# ------------

print("\n*** building model")

l_model = latent_model(l_config["e_dim"],w_config["z_dim"],l_config["h_dim"],l_config["n_linears"],
                       l_config["rnn_type"],l_config["n_RNN"],n_grains,classes,
                       args.conditional,learning_rate=args.learning_rate)
l_model.to(device)
l_model.init_beta(args.max_steps,args.tar_beta,beta_steps=args.beta_steps)
l_model.export_dir = os.path.join(tmp_dir,"exports") # to write export files

print("model running on device",l_model.device)
print("model hyper-parameters",l_model.hparams)

l_model.train()
for batch in train_latentloader:
    break
l_model.gradient_check(batch)


# ------------
# training
# ------------

print("\n*** start of STAGE 2 training")

time.sleep(10)

trainer.fit(l_model, train_latentloader, test_latentloader)

print("\n*** end of STAGE 2 training after #iter = ",l_model.acc_iter)


# ------------
# export
# ------------

l_model.to(device)
l_model.eval()

print("\n*** exporting embedding to audio reconstructions")

for batch in train_latentloader:
    break
export_embedding_to_audio_reconstructions(l_model,w_model,batch,trainset=True)
for batch in test_latentloader:
    break
export_embedding_to_audio_reconstructions(l_model,w_model,batch,trainset=False)

print("\n*** exporting random samples embedding to audio")

export_random_samples(l_model,w_model,l_model.export_dir,n_samples=10)

print("\n*** plotting embedding projections")

train_latents,train_labels,test_latents,test_labels = export_embeddings(l_model,train_latentloader,test_latentloader)

plot_embeddings(train_latents,train_labels,test_latents,test_labels,classes,l_model.export_dir)


# ------------
# misc.
# ------------

args = vars(args)
args["classes"] = classes # make sure the classes are saved in the sorted order used for training

np.save(os.path.join(tmp_dir,'argparse.npy'),args)
shutil.move(tmp_dir,os.path.join(curr_dir,args["out_dir"]))
shutil.rmtree(default_root_dir)
os.rename(os.path.join(curr_dir,args["out_dir"],"version_0"),default_root_dir)

# tensorboard --logdir 

