#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:43:08 2021

@author: adrienbitton
"""



import os
import soundfile as sf

import torch
from torch.utils.data import DataLoader



# ------------
# HIERARCHICAL AUTO-ENCODER (two-level embeddings)
# ------------


def export_hierarchical_audio_reconstructions(model,batch,trainset=True):
    if os.path.exists(model.export_dir) is False:
        os.makedirs(model.export_dir)
    with torch.no_grad():
        audio,labels = batch
        audio,labels = audio.to(model.device),labels.to(model.device)
        # forward through waveform and latent embeddings
        audio_output,_,_,_ = model.forward(audio, labels, sampling=False)
        audio_export = torch.cat((audio,audio_output),-1).cpu().numpy()
        for i in range(audio_output.shape[0]):
            if trainset:
                sf.write(os.path.join(model.export_dir,"audio_train_reconstruction_"+str(i)+".wav"),audio_export[i,:],model.w_model.hparams.sr)
            else:
                sf.write(os.path.join(model.export_dir,"audio_test_reconstruction_"+str(i)+".wav"),audio_export[i,:],model.w_model.hparams.sr)


def export_random_samples(model,export_dir,n_samples=10,temperature=1.):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    with torch.no_grad():
        for i,cl in enumerate(model.l_model.hparams.classes):
            rand_e = torch.randn((n_samples,model.l_model.hparams.e_dim)).to(model.device)
            rand_e = rand_e*temperature
            conds = torch.zeros(n_samples).to(model.device).long()+i
            audio_hat = model.decode(rand_e,conds)[0].view(-1).cpu().numpy()
            sf.write(os.path.join(export_dir,"random_samples_"+cl+".wav"),audio_hat,model.w_model.hparams.sr)


def compute_audio_to_embeddings(model,dataloader):
    tmploader = DataLoader(dataloader.dataset, batch_size=5, shuffle=False, drop_last=False)
    dataset_latents = []
    dataset_labels = []
    for _,batch in enumerate(tmploader):
        with torch.no_grad():
            audio,labels = batch
            mu = model.encode(audio.to(model.device),sampling=False)[0]["mu"].cpu()
            dataset_latents.append(mu)
            dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    dataset_labels = torch.cat(dataset_labels,0)
    print("* exported dataset sizes",dataset_latents.shape,dataset_labels.shape)
    return dataset_latents,dataset_labels

def export_audio_to_embeddings(model,train_dataloader,test_dataloader):
    train_latents,train_labels = compute_audio_to_embeddings(model,train_dataloader)
    test_latents,test_labels = compute_audio_to_embeddings(model,test_dataloader)
    return train_latents,train_labels,test_latents,test_labels



# if __name__ == "__main__":
    
    
    