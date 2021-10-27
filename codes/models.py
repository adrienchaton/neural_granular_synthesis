#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:13:31 2021

@author: adrienbitton
"""



import numpy as np
import math
from scipy import signal

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import nn
from torchaudio.transforms import Spectrogram,MelSpectrogram



# ------------
# VARIATIONAL AUTO-ENCODER N(0,1)
# ------------

def reparametrize(mu,logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

def compute_kld(mu,logvar):
    # TODO: add weighting of M/N = latent/input sizes
    mu = torch.flatten(mu, start_dim=1)
    logvar = torch.flatten(logvar, start_dim=1)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    return kld_loss


# ------------
# DDSP operations
# references: https://github.com/magenta/ddsp ; https://github.com/acids-ircam/ddsp_pytorch
# ------------

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

def noise_filtering(filter_coeffs,filter_window):
    N = filter_coeffs.shape[0]
    l_grain = (filter_coeffs.shape[1]-1)*2
    dtype = filter_coeffs.dtype
    # create impulse response
    filter_coeffs = torch.complex(filter_coeffs,torch.zeros_like(filter_coeffs))
    filter_ir = torch.fft.irfft(filter_coeffs)
    filter_ir = filter_ir*filter_window.unsqueeze(0).repeat(N,1)
    filter_ir = torch.fft.fftshift(filter_ir,dim=-1)
    # convolve with noise signal
    noise = torch.rand(N,l_grain, dtype=dtype, device=filter_coeffs.device)*2-1
    S_noise = torch.fft.rfft(noise,dim=1)
    S_filter = torch.fft.rfft(filter_ir,dim=1)
    S = torch.mul(S_noise,S_filter)
    audio = torch.fft.irfft(S)
    return audio

class spectral_distances(nn.Module):
    def __init__(self,stft_scales=[2048, 1024, 512, 256, 128], mel_scales=[2048, 1024], spec_power=1, mel_dist=True, log_dist=0, sr=16000, device="cpu"):
        super(spectral_distances, self).__init__()
        self.stft_scales = stft_scales
        self.mel_scales = mel_scales
        self.mel_dist = mel_dist
        self.log_dist = log_dist
        T_spec = []
        for scale in stft_scales:
            T_spec.append(Spectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,power=spec_power).to(device))
        self.T_spec = T_spec
        if mel_dist:
            print("\n*** training with MelSpectrogram distance")
            T_mel = []
            for scale in mel_scales:
                T_mel.append(MelSpectrogram(n_fft=scale,hop_length=scale//4,window_fn=torch.hann_window,sample_rate=sr,f_min=50.,n_mels=scale//4,power=spec_power).to(device))
            self.T_mel = T_mel
    
    def forward(self,x_inp,x_tar):
        loss = 0
        n_scales = 0
        for i,scale in enumerate(self.stft_scales):
            S_inp,S_tar = self.T_spec[i](x_inp),self.T_spec[i](x_tar)
            stft_dist = (S_inp-S_tar).abs().mean()
            loss = loss+stft_dist
            n_scales += 1
            if self.log_dist>0:
                loss = loss+(safe_log(S_inp)-safe_log(S_tar)).abs().mean()*self.log_dist
                n_scales += self.log_dist
        if self.mel_dist:
            for i,scale in enumerate(self.mel_scales):
                M_inp,M_tar = self.T_mel[i](x_inp),self.T_mel[i](x_tar)
                mel_dist = (M_inp-M_tar).abs().mean()
                loss = loss+mel_dist
                n_scales += 1
                if self.log_dist>0:
                    loss = loss+(safe_log(M_inp)-safe_log(M_tar)).abs().mean()*self.log_dist
                    n_scales += self.log_dist
        return loss/n_scales

def envelope_distance(x_inp,x_tar,n_fft=1024,log=True):
    env_inp = torch.stft(x_inp, n_fft, hop_length=n_fft//4, onesided=True, return_complex=False)
    env_inp = torch.mean(env_inp[:,:,:,0]**2+env_inp[:,:,:,1]**2,1)
    env_tar = torch.stft(x_tar, n_fft, hop_length=n_fft//4, onesided=True, return_complex=False)
    env_tar = torch.mean(env_tar[:,:,:,0]**2+env_tar[:,:,:,1]**2,1)
    if log:
        env_inp,env_tar = safe_log(env_inp),safe_log(env_tar)
    return (env_inp-env_tar).abs().mean()


# ------------
# WAVEFORM AUTO-ENCODER (grain-level embedding)
# ------------

class stride_conv(nn.Module):
    def __init__(self,kernel_size,in_channels,out_channels,stride):
        super(stride_conv, self).__init__()
        # kernel should be an odd number and stride an even number
        self.conv = nn.Sequential(nn.ReflectionPad1d(kernel_size//2),
                                  nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                                  nn.BatchNorm1d(out_channels),nn.LeakyReLU(0.2))
    
    def forward(self, x):
        # input and output of shape [bs,in_channels,L] --> [bs,out_channels,L//stride]
        return self.conv(x)


class residual_conv(nn.Module):
    def __init__(self, channels,n_blocks=3):
        super(residual_conv, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.Conv1d(channels, channels, kernel_size=3, dilation=3**i),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.BatchNorm1d(channels))
        for i in range(n_blocks)])
        self.shortcuts = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                      nn.BatchNorm1d(channels))
        for i in range(n_blocks)])
    
    def forward(self, x):
        # input and output of shape [bs,channels,L]
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x


class linear_block(nn.Module):
    def __init__(self, in_size,out_size,norm="BN"):
        super(linear_block, self).__init__()
        if norm=="BN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.BatchNorm1d(out_size),nn.LeakyReLU(0.2))
        if norm=="LN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.LayerNorm(out_size),nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.block(x)


class waveform_model(pl.LightningModule):
    def __init__(self,z_dim,h_dim,kernel_size,channels,n_convs,stride,n_linears,n_grains,hop_size,normalize_ola,pp_chans,pp_ker,l_grain=2048,sr=16000,learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.acc_iter = 0
        
        # fixed parameters
        
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.filter_size = l_grain//2+1
        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False)
        
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2]
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)
        
        self.slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False)
        
        self.ola_folder = nn.Fold((self.tar_l,1),(l_grain,1),stride=(hop_size,1))
        if normalize_ola:
            unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,self.tar_l,1)
            ola_divisor = self.ola_folder(unfolder(input_ones)).squeeze()
            self.ola_divisor = nn.Parameter(ola_divisor,requires_grad=False)
        
        # encoder parameters
        
        encoder_convs = [nn.Sequential(stride_conv(kernel_size,1,channels,stride),residual_conv(channels,n_blocks=3))]
        encoder_convs += [nn.Sequential(stride_conv(kernel_size,channels,channels,stride),residual_conv(channels,n_blocks=3)) for i in range(1,n_convs)]
        self.encoder_convs = nn.ModuleList(encoder_convs)
        
        self.flatten_size = int(channels*l_grain/(stride**n_convs))
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities
        
        # decoder parameters
        
        decoder_linears = [linear_block(z_dim,h_dim)]
        decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(h_dim,self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)
        
        self.post_pro = nn.Sequential(nn.Conv1d(pp_chans, 1, pp_ker, padding=pp_ker//2),nn.Softsign())
    
    def init_spectral_distances(self, stft_scales=[2048, 1024, 512, 256, 128], mel_scales=[2048, 1024], spec_power=1, mel_dist=True, log_dist=0., env_dist=0, device="cpu"):
        self.spec_dist = spectral_distances(stft_scales=stft_scales,mel_scales=mel_scales,spec_power=spec_power,mel_dist=mel_dist,log_dist=log_dist,sr=self.hparams.sr,device=device)
        self.env_dist = env_dist
    
    def init_beta(self,max_steps,tar_beta,beta_steps=1000):
        if self.continue_train:
            self.tar_beta = tar_beta
            self.beta = tar_beta
            print("\n*** setting fixed beta of ",self.beta)
        else:
            self.max_steps = max_steps
            self.tar_beta = tar_beta
            self.beta_steps = beta_steps # number of warmup steps over half max_steps
            self.warmup_start = int(0.1*max_steps)
            self.beta_step_size = int(max_steps/2/beta_steps)
            self.beta_step_val = tar_beta/beta_steps
            self.beta = 0
            print("\n*** setting beta warmup from 0 to ",tar_beta)
    
    def encode(self, x, print_shapes=False):
        # slicing input mini-batch of shape [bs,tar_l]
        mb_grains = F.conv1d(x.unsqueeze(1),self.slice_kernel,stride=self.hparams.hop_size,groups=1,bias=None)
        mb_grains = mb_grains.permute(0,2,1)
        # windowing input mb_grains of shape [bs,n_grains,l_grain]
        bs = mb_grains.shape[0]
        mb_grains = mb_grains*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        mb_grains = mb_grains.reshape(bs*self.hparams.n_grains,self.hparams.l_grain).unsqueeze(1)
        if print_shapes:
            print("input batch size",mb_grains.shape)
        # mb_grains of shape [bs*n_grains,1,l_grain]
        for conv in self.encoder_convs:
            mb_grains = conv(mb_grains)
            if print_shapes:
                print("output conv size",mb_grains.shape)
        mb_grains = mb_grains.view(-1,self.flatten_size)
        if print_shapes:
            print("flatten size",mb_grains.shape)
        h = self.encoder_linears(mb_grains)
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = reparametrize(mu,logvar)
        return {"z":z,"mu":mu,"logvar":logvar}
    
    def decode(self,z,n_grains=None,ola_windows=None,ola_folder=None,ola_divisor=None):
        filter_coeffs = self.decoder_linears(z)
        # filter_coeffs of shape [bs*n_grains,filter_size]
        filter_coeffs = mod_sigmoid(filter_coeffs)
        audio = noise_filtering(filter_coeffs,self.filter_window)
        # windowing from audio of shape [bs*n_grains,l_grain]
        if n_grains is None:
            audio = audio.reshape(-1,self.hparams.n_grains,self.hparams.l_grain)
        else:
            audio = audio.reshape(-1,n_grains,self.hparams.l_grain)
        bs = audio.shape[0]
        if ola_windows is None:
            audio = audio*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        else:
            audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))
        # overlap-add
        if ola_folder is None:
            audio_sum = self.ola_folder(audio.permute(0,2,1)).squeeze()
        else:
            audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()
        if self.hparams.normalize_ola:
            if ola_divisor is None:
                audio_sum = audio_sum/self.ola_divisor.unsqueeze(0).repeat(bs,1)
            else:
                audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
        # post-processing of audio_sum of shape [bs,tar_l]
        audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.hparams.pp_chans,1)).squeeze(1)
        return audio_sum

    def forward(self, audio, sampling=True):
        encoder_outputs = self.encode(audio)
        if sampling:
            audio_output = self.decode(encoder_outputs["z"])
        else:
            audio_output = self.decode(encoder_outputs["mu"])
        return audio_output,encoder_outputs

    def compute_losses(self, batch, beta):
        audio,labels = batch
        audio = audio.to(self.device)
        # forward
        audio_output,encoder_outputs = self.forward(audio, sampling=True)
        # compute losses
        spec_loss = self.spec_dist(audio_output,audio)
        if beta>0:
            kld_loss = compute_kld(encoder_outputs["mu"],encoder_outputs["logvar"])*beta
        else:
            kld_loss = 0
        if self.env_dist>0:
            env_loss = envelope_distance(audio_output,audio,n_fft=1024,log=True)*self.env_dist
        else:
            env_loss = 0
        return {"spec_loss":spec_loss,"kld_loss":kld_loss,"env_loss":env_loss}

    def training_step(self, batch, batch_idx):
        if not self.continue_train:
            if (self.acc_iter+1)%self.beta_step_size==0:
                if self.acc_iter<self.warmup_start:
                    self.beta = 0
                elif self.beta<self.tar_beta:
                    self.beta += self.beta_step_val
                    self.beta = np.min([self.beta,self.tar_beta])
                else:
                    self.beta = self.tar_beta
        losses = self.compute_losses(batch,self.beta)
        spec_loss,kld_loss,env_loss = losses["spec_loss"],losses["kld_loss"],losses["env_loss"]
        loss = spec_loss+kld_loss+env_loss
        self.log("waveform_train_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_train_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_train_kld_loss", kld_loss, on_step=False, on_epoch=True)
        self.log("waveform_beta_kld", self.beta, on_step=False, on_epoch=True)
        if self.env_dist>0:
            self.log("waveform_train_env_loss", env_loss)
        self.acc_iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch,self.beta)
        spec_loss,kld_loss,env_loss = losses["spec_loss"],losses["kld_loss"],losses["env_loss"]
        loss = spec_loss+kld_loss+env_loss
        self.log("waveform_test_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_test_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_test_kld_loss", kld_loss, on_step=False, on_epoch=True)
        if self.env_dist>0:
            self.log("waveform_test_env_loss", env_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.continue_train:
            print("*** setting fixed learning rate of",self.hparams.learning_rate)
            return opt
        else:
            lr_decay = 1e-2
            lr_scale = np.exp(np.log(lr_decay)/self.max_steps)
            print("*** setting exponential decay of learning rate with factor and final value to",lr_scale,self.hparams.learning_rate*lr_scale**self.max_steps)
            scheduler = {'scheduler':torch.optim.lr_scheduler.ExponentialLR(opt, lr_scale, verbose=False),'interval':'step'}
            # TODO: lr_scheduler may be best by stepping every epoch for ExponentialLR ?
            return [opt],[scheduler]

    def gradient_check(self, batch):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.env_dist>0:
            training_losses = ["spec_loss","env_loss","kld_loss"]
        else:
            training_losses = ["spec_loss","kld_loss"]
        for loss in training_losses:
            print("\n*** "+loss+" initial gradient check")
            if loss!="kld_loss":
                losses = self.compute_losses(batch,0.)
            else:
                losses = self.compute_losses(batch,self.tar_beta)
            opt.zero_grad()
            losses[loss].backward()
            tot_grad = 0
            named_p = self.named_parameters()
            for name, param in named_p:
                if param.grad is not None:
                    sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                    if sum_abs_paramgrad == 0:
                        print(name, "sum_abs_paramgrad==0")
                    else:
                        tot_grad += sum_abs_paramgrad
                else:
                    print(name, "param.grad is None")
            print("tot_grad = ", tot_grad)
        opt.zero_grad()


# ------------
# LATENT AUTO-ENCODER (series-level embedding)
# ------------

class latent_model(pl.LightningModule):
    def __init__(self,e_dim,z_dim,h_dim,n_linears,rnn_type,n_RNN,n_grains,classes,conditional,learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.acc_iter = 0
        
        # when len(classes)>1 and conditional=True then the decoder receives a one-hot vector of size len(classes)
        # TODO: replace one-hot condition with FiLM modulation ? or add Fader-Network regularization to the encoder
        if conditional is True and len(classes)>1:
            self.n_conds = len(classes)
            print("\n*** training latent VAE with class conditioning over",classes)
        else:
            self.n_conds = 0
        
        # encoder parameters
        
        encoder_z = [linear_block(z_dim,h_dim,norm="LN")]
        encoder_z += [linear_block(h_dim,h_dim,norm="LN") for i in range(1,n_linears)]
        self.encoder_z = nn.Sequential(*encoder_z)
        
        if rnn_type=="GRU":
            self.encoder_rnn = nn.GRU(h_dim,h_dim,num_layers=n_RNN,batch_first=True)
        if rnn_type=="LSTM":
            self.encoder_rnn = nn.LSTM(h_dim,h_dim,num_layers=n_RNN,batch_first=True)
        
        encoder_e = [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        encoder_e += [linear_block(h_dim,e_dim)]
        self.encoder_e = nn.Sequential(*encoder_e)
        
        self.mu = nn.Linear(e_dim,e_dim)
        self.logvar = nn.Sequential(nn.Linear(e_dim,e_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities
        
        # decoder parameters
        
        decoder_e = [linear_block(e_dim+self.n_conds,h_dim)] # global conditioning before the RNN
        decoder_e += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        self.decoder_e = nn.Sequential(*decoder_e)
        
        if rnn_type=="GRU":
            self.decoder_rnn = nn.GRU(h_dim,h_dim,num_layers=n_RNN,batch_first=True)
        if rnn_type=="LSTM":
            self.decoder_rnn = nn.LSTM(h_dim,h_dim,num_layers=n_RNN,batch_first=True)
        
        decoder_z = [linear_block(h_dim+self.n_conds,h_dim,norm="LN")] # granular conditioning after the RNN
        decoder_z += [linear_block(h_dim,h_dim,norm="LN") for i in range(1,n_linears)]
        decoder_z += [nn.Linear(h_dim,z_dim)]
        self.decoder_z = nn.Sequential(*decoder_z)
        
    def init_beta(self,max_steps,tar_beta,beta_steps=1000):
        self.max_steps = max_steps
        self.tar_beta = tar_beta
        self.beta_steps = beta_steps # number of warmup steps over half max_steps
        self.warmup_start = int(0.1*max_steps)
        self.beta_step_size = int(max_steps/2/beta_steps)
        self.beta_step_val = tar_beta/beta_steps
        self.beta = 0
        print("\n*** setting beta warmup from 0 to ",tar_beta)
    
    def encode(self, z):
        # input series of latents of shape [bs,n_grains,z_dim]
        h = self.encoder_z(z)
        _,h_n = self.encoder_rnn(h)
        if self.hparams.rnn_type=="LSTM":
            h_n = h_n[0] # we ommit the additional LSTM cell state
            # using the last cell state to init the decoder prevents from random sampling (without encoder outputs)
        h = self.encoder_e(h_n[-1,:,:])
        mu = self.mu(h)
        logvar = self.logvar(h)
        e = reparametrize(mu,logvar)
        return {"e":e,"mu":mu,"logvar":logvar}
    
    def decode(self,e,conds):
        # input embedding of shape [N,e_dim] and conds of shape [N] (long)
        if self.n_conds>0:
            conds = F.one_hot(conds, num_classes=self.n_conds)
            e = torch.cat((e,conds),1)
        h = self.decoder_e(e)
        h = h.unsqueeze(1).repeat(1,self.hparams.n_grains,1).contiguous()
        # otherwise could use an auto-regressive approach if mean seaking
        # e.g. https://stackoverflow.com/questions/65205506/lstm-autoencoder-problems
        h,_ = self.decoder_rnn(h)
        if self.n_conds>0:
            conds = conds.unsqueeze(1).repeat(1,self.hparams.n_grains,1).contiguous()
            h = torch.cat((h,conds),2)
        z = self.decoder_z(h)
        return z
    
    def forward(self, z, conds, sampling=True):
        encoder_outputs = self.encode(z)
        if sampling:
            z_hat = self.decode(encoder_outputs["e"],conds)
        else:
            z_hat = self.decode(encoder_outputs["mu"],conds)
        return z_hat,encoder_outputs
    
    def compute_losses(self, batch, beta):
        z,conds = batch
        z,conds = z.to(self.device),conds.to(self.device)
        # forward
        z_hat,encoder_outputs = self.forward(z,conds, sampling=True)
        # compute losses
        rec_loss = F.mse_loss(z_hat,z) # we train with a deterministic output
        # TODO: compare with gaussian output and KLD distance ?
        if beta>0:
            kld_loss = compute_kld(encoder_outputs["mu"],encoder_outputs["logvar"])*beta
        else:
            kld_loss = 0
        return {"rec_loss":rec_loss,"kld_loss":kld_loss}

    def training_step(self, batch, batch_idx):
        if (self.acc_iter+1)%self.beta_step_size==0:
            if self.acc_iter<self.warmup_start:
                self.beta = 0
            elif self.beta<self.tar_beta:
                self.beta += self.beta_step_val
                self.beta = np.min([self.beta,self.tar_beta])
            else:
                self.beta = self.tar_beta
        losses = self.compute_losses(batch,self.beta)
        rec_loss,kld_loss = losses["rec_loss"],losses["kld_loss"]
        loss = rec_loss+kld_loss
        self.log("latent_train_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("latent_train_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("latent_train_kld_loss", kld_loss, on_step=False, on_epoch=True)
        self.log("latent_beta_kld", self.beta, on_step=False, on_epoch=True)
        self.acc_iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch,self.beta)
        rec_loss,kld_loss = losses["rec_loss"],losses["kld_loss"]
        loss = rec_loss+kld_loss
        self.log("latent_test_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("latent_test_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("latent_test_kld_loss", kld_loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_decay = 1e-2
        lr_scale = np.exp(np.log(lr_decay)/self.max_steps)
        print("*** setting exponential decay of learning rate with factor and final value to",lr_scale,self.hparams.learning_rate*lr_scale**self.max_steps)
        scheduler = {'scheduler':torch.optim.lr_scheduler.ExponentialLR(opt, lr_scale, verbose=False),'interval':'step'}
        return [opt],[scheduler]

    def gradient_check(self, batch):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        for loss in ["rec_loss","kld_loss"]:
            print("\n*** "+loss+" initial gradient check")
            if loss!="kld_loss":
                losses = self.compute_losses(batch,0.)
            else:
                losses = self.compute_losses(batch,self.tar_beta)
            opt.zero_grad()
            losses[loss].backward()
            tot_grad = 0
            named_p = self.named_parameters()
            for name, param in named_p:
                if param.grad is not None:
                    sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                    if sum_abs_paramgrad == 0:
                        print(name, "sum_abs_paramgrad==0")
                    else:
                        tot_grad += sum_abs_paramgrad
                else:
                    print(name, "param.grad is None")
            print("tot_grad = ", tot_grad)
        opt.zero_grad()


# ------------
# HIERARCHICAL AUTO-ENCODER (two-level embeddings)
# ------------

class hierarchical_model(pl.LightningModule):
    def __init__(self,w_ckpt_file="w_ckpt_file",w_yaml_file="w_yaml_file",l_ckpt_file="l_ckpt_file",l_yaml_file="l_yaml_file",learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.acc_iter = 0
        
        self.w_model = waveform_model.load_from_checkpoint(checkpoint_path=w_ckpt_file,hparams_file=w_yaml_file,map_location='cpu')
        self.l_model = latent_model.load_from_checkpoint(checkpoint_path=l_ckpt_file,hparams_file=l_yaml_file,map_location='cpu')
    
    def init_spectral_distances(self, w_config, device="cpu"):
        self.spec_dist = spectral_distances(stft_scales=w_config["stft_scales"],mel_scales=w_config["mel_scales"],spec_power=w_config["spec_power"],
                                        mel_dist=w_config["mel_dist"],log_dist=w_config["log_dist"],sr=self.w_model.hparams.sr,device=device)
        self.env_dist = w_config["env_dist"]
    
    def init_beta(self,w_args,l_args,w_beta=0.,l_beta=0.):
        if w_beta==0.:
            self.w_beta = w_args["tar_beta"]
        else:
            self.w_beta = w_beta
        if l_beta==0.:
            self.l_beta = l_args["tar_beta"]
        else:
            self.l_beta = l_beta
        print("\n*** setting fixed beta for bottom and top KLD of ",self.w_beta,self.l_beta)
    
    def encode(self, x, sampling=True):
        w_encoder_outputs = self.w_model.encode(x)
        if sampling:
            l_encoder_outputs = self.l_model.encode(w_encoder_outputs["z"].reshape(-1,self.l_model.hparams.n_grains,self.l_model.hparams.z_dim))
        else:
            l_encoder_outputs = self.l_model.encode(w_encoder_outputs["mu"].reshape(-1,self.l_model.hparams.n_grains,self.l_model.hparams.z_dim))
        return l_encoder_outputs,w_encoder_outputs
    
    def decode(self,e,conds):
        z_hat = self.l_model.decode(e,conds).reshape(-1,self.l_model.hparams.z_dim)
        audio_output = self.w_model.decode(z_hat)
        return audio_output,z_hat
    
    def forward(self, audio, conds, sampling=True):
        l_encoder_outputs,w_encoder_outputs = self.encode(audio, sampling=sampling)
        if sampling:
            audio_output,z_hat = self.decode(l_encoder_outputs["e"],conds)
        else:
            audio_output,z_hat = self.decode(l_encoder_outputs["mu"],conds)
        return audio_output,z_hat,l_encoder_outputs,w_encoder_outputs
    
    def compute_losses(self, batch):
        audio,labels = batch
        audio,labels = audio.to(self.device),labels.to(self.device)
        # forward
        audio_output,z_hat,l_encoder_outputs,w_encoder_outputs = self.forward(audio, labels, sampling=True)
        # compute losses
        spec_loss = self.spec_dist(audio_output,audio)
        w_kld_loss = compute_kld(w_encoder_outputs["mu"],w_encoder_outputs["logvar"])*self.w_beta
        if self.env_dist>0:
            env_loss = envelope_distance(audio_output,audio,n_fft=1024,log=True)*self.env_dist
        else:
            env_loss = 0
        l_kld_loss = compute_kld(l_encoder_outputs["mu"],l_encoder_outputs["logvar"])*self.l_beta
        l_rec_loss = F.mse_loss(z_hat,w_encoder_outputs["z"])
        return {"spec_loss":spec_loss,"w_kld_loss":w_kld_loss,"env_loss":env_loss,"l_rec_loss":l_rec_loss,"l_kld_loss":l_kld_loss}
    
    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        spec_loss,w_kld_loss,env_loss,l_rec_loss,l_kld_loss = losses["spec_loss"],losses["w_kld_loss"],losses["env_loss"],losses["l_rec_loss"],losses["l_kld_loss"]
        loss = spec_loss+w_kld_loss+env_loss+l_rec_loss+l_kld_loss
        self.log("train_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_train_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_train_kld_loss", w_kld_loss, on_step=False, on_epoch=True)
        if self.env_dist>0:
            self.log("waveform_train_env_loss", env_loss)
        self.log("latent_train_rec_loss", l_rec_loss, on_step=False, on_epoch=True)
        self.log("latent_train_kld_loss", l_kld_loss, on_step=False, on_epoch=True)
        self.acc_iter += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        spec_loss,w_kld_loss,env_loss,l_rec_loss,l_kld_loss = losses["spec_loss"],losses["w_kld_loss"],losses["env_loss"],losses["l_rec_loss"],losses["l_kld_loss"]
        loss = spec_loss+w_kld_loss+env_loss+l_rec_loss+l_kld_loss
        self.log("test_tot_loss", loss, on_step=False, on_epoch=True)
        self.log("waveform_test_spec_loss", spec_loss, on_step=False, on_epoch=True)
        self.log("waveform_test_kld_loss", w_kld_loss, on_step=False, on_epoch=True)
        if self.env_dist>0:
            self.log("waveform_test_env_loss", env_loss)
        self.log("latent_test_rec_loss", l_rec_loss, on_step=False, on_epoch=True)
        self.log("latent_test_kld_loss", l_kld_loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        print("*** setting fixed learning rate of",self.hparams.learning_rate)
        return opt
    
    def gradient_check(self, batch):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.env_dist>0:
            training_losses = ["spec_loss","w_kld_loss","env_loss","l_rec_loss","l_kld_loss"]
        else:
            training_losses = ["spec_loss","w_kld_loss","l_rec_loss","l_kld_loss"]
        for loss in training_losses:
            print("\n*** "+loss+" initial gradient check")
            losses = self.compute_losses(batch)
            opt.zero_grad()
            losses[loss].backward()
            tot_grad = 0
            named_p = self.named_parameters()
            for name, param in named_p:
                if param.grad is not None:
                    sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                    if sum_abs_paramgrad == 0:
                        print(name, "sum_abs_paramgrad==0")
                    else:
                        tot_grad += sum_abs_paramgrad
                else:
                    print(name, "param.grad is None")
            print("tot_grad = ", tot_grad)
        opt.zero_grad()



if __name__ == "__main__":
    
    # examples of creating model configurations
    
    """
    import json
    l_config = dict()
    l_config["e_dim"] = 256
    l_config["rnn_type"] = "LSTM"
    l_config["n_RNN"] = 1
    l_config["h_dim"] = 512
    l_config["n_linears"] = 2
    with open("./configs/l_E256_1LSTM.json", 'w') as f:
        json.dump(l_config, f, sort_keys=True, indent=4)
    """
    
    
    """
    import json
    w_config = dict()
    w_config["sr"] = 22050
    w_config["l_grain"] = 2048
    w_config["silent_reject"] = [0.2,0.2] # first value is minimum peak amplitude, second is minimum non-silent length ratio to target length (0=False)
    # or simpler rejection sampling e.g. https://github.com/NVIDIA/waveglow/issues/155#issuecomment-531029586 ?
    w_config["amplitude_norm"] = False # amplitude normalization of training files
    w_config["normalize_ola"] = True # normalization of the overlap-add output of the model
    w_config["mel_dist"] = True
    w_config["log_dist"] = 0. # scale factor of the log-magnitude distances (0=False)
    w_config["spec_power"] = 1
    w_config["env_dist"] = 0 # scale factor of the envelope distance (0=False)
    w_config["stft_scales"] = [2048, 1024, 512, 256]
    w_config["mel_scales"] = [2048, 1024]
    
    w_config["z_dim"] = 128
    w_config["h_dim"] = 512
    w_config["kernel_size"] = 9
    w_config["channels"] = 128
    w_config["n_convs"] = 3
    w_config["stride"] = 4
    w_config["n_linears"] = 3
    w_config["pp_chans"] = 5
    w_config["pp_ker"] = 65
    with open("./configs/w_22k_L2048_Reject0202_normola.json", 'w') as f:
        json.dump(w_config, f, sort_keys=True, indent=4)
    """
    
    