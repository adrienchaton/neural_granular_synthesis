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

from models import hierarchical_model
from utils_stage1 import make_audio_dataloaders
from utils_stage2 import plot_embeddings
from utils_stage3 import export_hierarchical_audio_reconstructions,export_random_samples,export_audio_to_embeddings
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
parser.add_argument('--latent_mname', default="test_latent_model", type=str)
parser.add_argument('--waveform_mname', default="test_waveform_model", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--learning_rate', default=2e-6, type=float) # here is the fixed learning rate at the end of the decay of the sub-network pretraining
parser.add_argument('--w_beta', default=0., type=float)
parser.add_argument('--l_beta', default=0., type=float)
parser.add_argument('--max_steps', default=100000, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--gpus', default=1, type=int)
parser.add_argument('--precision', default=32, type=int)
parser.add_argument('--profiler', action='store_true')
parser.add_argument('--out_dir', default="outputs", type=str)
args = parser.parse_args()

args.mname = args.waveform_mname+"__"+args.latent_mname+"__finetuned"
if args.w_beta>0.:
    args.mname += "_wbeta"+str(args.w_beta)
if args.l_beta>0.:
    args.mname += "_lbeta"+str(args.l_beta)
args.latent_mname = args.waveform_mname+"__"+args.latent_mname

default_root_dir = os.path.join(curr_dir, args.out_dir, args.mname)
print("writing outputs into default_root_dir",default_root_dir)

# lighting is writting output files in default_root_dir/lightning_logs/version_0/
tmp_dir = os.path.join(default_root_dir,"lightning_logs","version_0")


###############################################################################
## STAGE 1 & 2: loading configuration aof waveform and latent VAEs + creating audio dataset
###############################################################################

print("\n*** loading of pretrained waveform VAE from",os.path.join(curr_dir, args.out_dir, args.waveform_mname))

w_args = np.load(os.path.join(curr_dir, args.out_dir, args.waveform_mname, "argparse.npy"),allow_pickle=True).item()
with open(w_args["w_config"]) as json_file:
    w_config = json.load(json_file)
w_ckpt_file = sorted(glob.glob(os.path.join(curr_dir, args.out_dir, args.waveform_mname, "checkpoints", "*.ckpt")))[-1]
w_yaml_file = os.path.join(curr_dir, args.out_dir, args.waveform_mname, "hparams.yaml")

print("\n*** loading of pretrained latent VAE from",os.path.join(curr_dir, args.out_dir, args.latent_mname))

l_args = np.load(os.path.join(curr_dir, args.out_dir, args.latent_mname, "argparse.npy"),allow_pickle=True).item()
with open(l_args["l_config"]) as json_file:
    l_config = json.load(json_file)
l_ckpt_file = sorted(glob.glob(os.path.join(curr_dir, args.out_dir, args.latent_mname, "checkpoints", "*.ckpt")))[-1]
l_yaml_file = os.path.join(curr_dir, args.out_dir, args.latent_mname, "hparams.yaml")

print("\n*** loading audio data")

train_dataloader,test_dataloader,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(w_args["data_dir"],
                            w_args["classes"],w_config["sr"],w_config["silent_reject"],w_config["amplitude_norm"],args.batch_size,tar_l=w_args["tar_l"],l_grain=w_config["l_grain"],
                            high_pass_freq=50.,num_workers=args.num_workers)


###############################################################################
## STAGE 2: training latent VAE
###############################################################################

print("\n*** STAGE 3: fine-tuning of waveform and latent VAEs")

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(max_steps=args.max_steps,check_val_every_n_epoch=1,
                     gpus=args.gpus,precision=args.precision,benchmark=True,
                     default_root_dir=default_root_dir,profiler=args.profiler,
                     progress_bar_refresh_rate=50,callbacks=[lr_monitor])


# ------------
# model
# ------------

print("\n*** building model")

model = hierarchical_model(w_ckpt_file=w_ckpt_file,w_yaml_file=w_yaml_file,
                       l_ckpt_file=l_ckpt_file,l_yaml_file=l_yaml_file,
                       learning_rate=args.learning_rate)
model.to(device)
model.init_beta(w_args,l_args,w_beta=args.w_beta,l_beta=args.l_beta)
model.init_spectral_distances(w_config, device=device)
model.export_dir = os.path.join(tmp_dir,"exports") # to write export files

print("model running on device",model.device)
print("model hyper-parameters",model.hparams)

model.train()
for batch in train_dataloader:
    break
model.gradient_check(batch)


# ------------
# training
# ------------

print("\n*** start of STAGE 3 training")

time.sleep(10)

trainer.fit(model, train_dataloader, test_dataloader)

print("\n*** end of STAGE 3 training after #iter = ",model.acc_iter)


# ------------
# export
# ------------

model.to(device)
model.eval()

print("\n*** exporting hierarchical audio reconstructions")

for batch in train_dataloader:
    break
export_hierarchical_audio_reconstructions(model,batch,trainset=True)
for batch in test_dataloader:
    break
export_hierarchical_audio_reconstructions(model,batch,trainset=False)

print("\n*** exporting random samples embedding to audio")

export_random_samples(model,model.export_dir,n_samples=10)

print("\n*** plotting embedding projections")

train_latents,train_labels,test_latents,test_labels = export_audio_to_embeddings(model,train_dataloader,test_dataloader)

plot_embeddings(train_latents,train_labels,test_latents,test_labels,classes,model.export_dir)


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

