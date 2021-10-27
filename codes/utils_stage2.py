#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:43:08 2021

@author: adrienbitton
"""



import os
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader



# ------------
# LATENT AUTO-ENCODER (series-level embedding)
# ------------

def make_latent_dataloaders(train_latents,train_labels,test_latents,test_labels,batch_size,num_workers=2):
    train_dataset = torch.utils.data.TensorDataset(train_latents,train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_latents,test_labels)
    # shapes are latents = [N,n_grains,z_dim] ; labels = [N]
    print("\n* latent train/test sizes",len(train_dataset),len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    return train_dataloader,test_dataloader


def export_embedding_to_audio_reconstructions(l_model,w_model,batch,trainset=True):
    if os.path.exists(l_model.export_dir) is False:
        os.makedirs(l_model.export_dir)
    with torch.no_grad():
        z,conds = batch
        z,conds = z.to(l_model.device),conds.to(l_model.device)
        # forward through latent embedding
        z_hat,_ = l_model.forward(z,conds, sampling=False)
        # reshape as minibatch of individual grains of shape [bs*n_grains,z_dim]
        z,z_hat = z.reshape(-1,w_model.hparams.z_dim),z_hat.reshape(-1,w_model.hparams.z_dim)
        # export reconstruction by pretrained waveform model and by embedding + waveform models
        audio,audio_hat = w_model.decode(z),w_model.decode(z_hat)
        audio_export = torch.cat((audio,audio_hat),-1).cpu().numpy()
        for i in range(audio_hat.shape[0]):
            if trainset:
                sf.write(os.path.join(l_model.export_dir,"embedding_to_audio_train_reconstruction_"+str(i)+".wav"),audio_export[i,:],w_model.hparams.sr)
            else:
                sf.write(os.path.join(l_model.export_dir,"embedding_to_audio_test_reconstruction_"+str(i)+".wav"),audio_export[i,:],w_model.hparams.sr)


def export_random_samples(l_model,w_model,export_dir,n_samples=10,temperature=1.):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    with torch.no_grad():
        for i,cl in enumerate(l_model.hparams.classes):
            rand_e = torch.randn((n_samples,l_model.hparams.e_dim)).to(l_model.device)
            rand_e = rand_e*temperature
            conds = torch.zeros(n_samples).to(l_model.device).long()+i
            z_hat = l_model.decode(rand_e,conds).reshape(-1,w_model.hparams.z_dim)
            audio_hat = w_model.decode(z_hat).view(-1).cpu().numpy()
            sf.write(os.path.join(export_dir,"random_samples_"+cl+".wav"),audio_hat,w_model.hparams.sr)


def compute_embeddings(l_model,dataloader):
    tmploader = DataLoader(dataloader.dataset, batch_size=5, shuffle=False, drop_last=False)
    dataset_latents = []
    dataset_labels = []
    for _,batch in enumerate(tmploader):
        with torch.no_grad():
            z,labels = batch
            mu = l_model.encode(z.to(l_model.device))["mu"].cpu()
            dataset_latents.append(mu)
            dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    dataset_labels = torch.cat(dataset_labels,0)
    print("* exported dataset sizes",dataset_latents.shape,dataset_labels.shape)
    return dataset_latents,dataset_labels

def export_embeddings(l_model,train_latentloader,test_latentloader):
    train_latents,train_labels = compute_embeddings(l_model,train_latentloader)
    test_latents,test_labels = compute_embeddings(l_model,test_latentloader)
    return train_latents,train_labels,test_latents,test_labels

def plot_embeddings(train_latents,train_labels,test_latents,test_labels,classes,export_dir):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    train_latents = train_latents.numpy()
    train_labels = train_labels.numpy().astype(str)
    test_latents = test_latents.numpy()
    test_labels = test_labels.numpy().astype(str)
    for i,c in enumerate(classes):
        train_labels[np.where(train_labels==str(i))] = c
        test_labels[np.where(test_labels==str(i))] = c
    pca = PCA(n_components=2)
    pca.fit(train_latents)
    train_latents = pca.transform(train_latents)
    test_latents = pca.transform(test_latents)
    # TODO: shuffle samples for better plotting
    sns.scatterplot(x=train_latents[:,0], y=train_latents[:,1], hue=train_labels, s=1)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(export_dir,"embedding_scatter_trainset.pdf"))
    plt.close("all")
    sns.scatterplot(x=test_latents[:,0], y=test_latents[:,1], hue=test_labels, s=1)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(export_dir,"embedding_scatter_testset.pdf"))
    plt.close("all")



# if __name__ == "__main__":
    
    
    