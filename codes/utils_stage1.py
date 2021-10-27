#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:43:08 2021

@author: adrienbitton
"""



import glob
import os
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import librosa

import torch
import torchaudio
from torch.utils.data import DataLoader, random_split



# ------------
# WAVEFORM AUTO-ENCODER (grain-level embedding)
# ------------

def make_audio_dataloaders(data_dir,classes,sr,silent_reject,amplitude_norm,batch_size,tar_l=1.1,l_grain=2048,high_pass_freq=50,num_workers=2):
    
    hop_ratio = 0.25 # hard-coded along with n_grains formula
    
    hop_size = int(hop_ratio*l_grain)
    tar_l = int(tar_l*sr)
    print("cropping from/to lengths",tar_l,tar_l//l_grain*l_grain)
    tar_l = int(tar_l//l_grain*l_grain)
    
    print("# non-overlapping grains",tar_l//l_grain)
    n_grains = 4*(tar_l//l_grain)-3
    print("# overlapping grains",n_grains)
    
    classes = sorted(classes)
    train_datasets = []
    test_datasets = []
    
    for i,drum in enumerate(classes):
        files = glob.glob(data_dir+drum+"/*.wav")
        print("\n*** class and # files",drum,len(files))
        audios = []
        labels = []
        n_rejected = 0
        for file in files:
            reject = 0
            data, samplerate = sf.read(file)
            if len(data.shape)>1:
                data = data.swapaxes(1, 0)
                data = librosa.to_mono(data)
            if samplerate!=sr:
                data = librosa.resample(data, samplerate, sr)
            
            data -= np.mean(data)
            if silent_reject[0]!=0 and np.max(np.abs(data))<silent_reject[0]:
                reject = 1 # peak amplitude is too low
            trim_pos = librosa.effects.trim(data, top_db=60, frame_length=1024, hop_length=128)[1]
            if silent_reject[1]!=0 and (trim_pos[1]-trim_pos[0])<silent_reject[1]*tar_l:
                reject = 1 # non-silent length is too low
            
            if reject==0:
                if len(data)<tar_l:
                    data = np.concatenate((data,np.zeros((tar_l-len(data)))))
                else:
                    data = data[:tar_l]
                
                data = torchaudio.functional.highpass_biquad(torch.from_numpy(data),sr,high_pass_freq).numpy()
                
                if amplitude_norm or np.max(np.abs(data))>=1:
                    data /= np.max(np.abs(data))
                    data *= 0.9
                
                audios.append(data)
                labels.append(i)
            else:
                n_rejected += 1
        
        print("n_rejected = ",n_rejected)
        
        audios = torch.from_numpy(np.stack(audios,axis=0)).float()
        labels = torch.from_numpy(np.stack(labels,axis=0)).long()
        print("* dataset sizes",audios.shape,labels.shape)
        
        n_drum = len(labels)
        n_train = int(n_drum*0.85)
        print("* split sizes",n_train, n_drum-n_train)
        dataset = torch.utils.data.TensorDataset(audios,labels)
        train_dataset,test_dataset = random_split(dataset, [n_train, n_drum-n_train])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    print("\n* final train/test sizes",len(train_dataset),len(test_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    
    return train_dataloader,test_dataloader,tar_l,n_grains,l_grain,hop_size,classes


def export_audio_reconstructions(w_model,batch,trainset=True):
    if os.path.exists(w_model.export_dir) is False:
        os.makedirs(w_model.export_dir)
    with torch.no_grad():
        audio,labels = batch
        audio = audio.to(w_model.device)
        audio_output,_ = w_model.forward(audio, sampling=False)
        audio_export = torch.cat((audio,audio_output),-1).cpu().numpy()
        for i in range(audio_output.shape[0]):
            if trainset:
                sf.write(os.path.join(w_model.export_dir,"waveform_train_reconstruction_"+str(i)+".wav"),audio_export[i,:],w_model.hparams.sr)
            else:
                sf.write(os.path.join(w_model.export_dir,"waveform_test_reconstruction_"+str(i)+".wav"),audio_export[i,:],w_model.hparams.sr)


def compute_latents(w_model,dataloader):
    tmploader = DataLoader(dataloader.dataset, batch_size=5, shuffle=False, drop_last=False)
    dataset_latents = []
    dataset_labels = []
    for _,batch in enumerate(tmploader):
        with torch.no_grad():
            audio,labels = batch
            bs = audio.shape[0]
            mu = w_model.encode(audio.to(w_model.device))["mu"].cpu()
            # mu of shape [bs*n_grains,z_dim]
            mu = mu.reshape(bs,w_model.hparams.n_grains,w_model.hparams.z_dim)
            dataset_latents.append(mu)
            dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    dataset_labels = torch.cat(dataset_labels,0)
    print("* exported dataset sizes",dataset_latents.shape,dataset_labels.shape)
    return dataset_latents,dataset_labels

def export_latents(w_model,train_dataloader,test_dataloader):
    train_latents,train_labels = compute_latents(w_model,train_dataloader)
    test_latents,test_labels = compute_latents(w_model,test_dataloader)
    return train_latents,train_labels,test_latents,test_labels

def plot_latents(train_latents,train_labels,test_latents,test_labels,classes,export_dir):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    n_grains = train_latents.shape[1]
    z_dim = train_latents.shape[2]
    train_latents = train_latents.view(-1,z_dim).numpy()
    train_labels = train_labels.unsqueeze(-1).repeat(1,n_grains).view(-1).numpy().astype(str)
    test_latents = test_latents.view(-1,z_dim).numpy()
    test_labels = test_labels.unsqueeze(-1).repeat(1,n_grains).view(-1).numpy().astype(str)
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
    plt.savefig(os.path.join(export_dir,"latent_scatter_trainset.pdf"))
    plt.close("all")
    sns.scatterplot(x=test_latents[:,0], y=test_latents[:,1], hue=test_labels, s=1)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(export_dir,"latent_scatter_testset.pdf"))
    plt.close("all")



# if __name__ == "__main__":
    