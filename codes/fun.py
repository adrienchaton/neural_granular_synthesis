#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:33:52 2021

@author: adrienbitton
"""



from argparse import ArgumentParser
import os
import glob
import numpy as np
import soundfile as sf
from scipy import signal

import torch
from torch import nn

from models import hierarchical_model



def make_interp(model,tar_l,temperature=1.):
    hop_ratio = 0.25 # hard-coded along with n_grains formula
    l_grain = model.w_model.hparams.l_grain
    sr = model.w_model.hparams.sr
    
    hop_size = int(hop_ratio*l_grain)
    tar_l = int(tar_l*sr)
    print("cropping from/to lengths",tar_l,tar_l//l_grain*l_grain)
    tar_l = int(tar_l//l_grain*l_grain)
    
    print("# non-overlapping grains",tar_l//l_grain)
    n_grains = 4*(tar_l//l_grain)-3
    print("# overlapping grains",n_grains)
    
    with torch.no_grad():
        # compute overlap-add parameters (not-learnable)
        # in order to generate overlap-add interpolations at different length thant the training n_grains
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2]
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2]
        ola_folder = nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
        unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
        input_ones = torch.ones(1,1,tar_l,1)
        ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
        # linear interpolation in the grain latent space
        z_0,z_1 = torch.randn((1,model.w_model.hparams.z_dim))*temperature,torch.randn((1,model.w_model.hparams.z_dim))*temperature
        alpha = np.linspace(0,1,num=n_grains,endpoint=True)
        z_interp = torch.cat([z_0*(1-a)+z_1*a for a in alpha],0)
        audio_interp = model.w_model.decode(z_interp,n_grains=n_grains,ola_windows=ola_windows,ola_folder=ola_folder,ola_divisor=ola_divisor).squeeze(0).numpy()
    return audio_interp


def sample(model,cl_idx,temperature=1.):
    with torch.no_grad():
        rand_e = torch.randn((1,model.l_model.hparams.e_dim))
        rand_e = rand_e*temperature
        conds = torch.zeros(1).long()+cl_idx
        audio = model.decode(rand_e,conds)[0].view(-1).numpy()
    return audio


def scale_audio(audio,amplitude):
    audio -= np.mean(audio)
    audio /= np.max(np.abs(audio))
    audio *= amplitude
    return audio


def generate_loop(model,loop_len,drum_dict,temperature=1.):
    loop_dict = dict()
    for cl in drum_dict:
        audio_loop = np.zeros(loop_len)
        cl_idx = model.l_model.hparams.classes.index(cl)
        events = drum_dict[cl]
        for onset,amplitude in events:
            audio = sample(model,cl_idx,temperature=temperature)
            audio = scale_audio(audio,amplitude)
            start_sample = int(onset*loop_len)
            end_sample = np.min([start_sample+len(audio),loop_len])
            audio_loop[start_sample:end_sample] += audio[:end_sample-start_sample]
        if np.max(np.abs(audio_loop))>0.95:
            audio_loop = scale_audio(audio_loop,0.95)
        loop_dict[cl] = audio_loop
    output_loop = np.zeros(loop_len)
    for cl in drum_dict:
        output_loop += loop_dict[cl]
    if np.max(np.abs(output_loop))>0.95:
        output_loop = scale_audio(output_loop,0.95)
    return output_loop,loop_dict


if __name__ == "__main__":
    
    # check issues in mostly silent interpolation ?
    # assert if fine-tuning (stage 3) degrades the model generative performances
    
    parser = ArgumentParser()
    parser.add_argument('--latent_mname', default="embedding_cond_l_E256_1LSTM", type=str)
    parser.add_argument('--waveform_mname', default="fulldrums_22k_L2048_Rej0202_normola", type=str)
    # parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--finetuned', default=0, type=int)
    parser.add_argument('--model_dir', default="outputs", type=str)
    parser.add_argument('--export_dir', default="outputs/generations", type=str)
    parser.add_argument('--samples_id', default=0, type=int)
    parser.add_argument('--temperature', default=1., type=float)
    parser.add_argument('--interp_len', default=4., type=float)
    args = parser.parse_args()
    
    if os.path.exists(args.export_dir) is False:
        os.makedirs(args.export_dir)
    
    # loading pretrained models: either using finetuned checkpoint of hierarchical_model
    # or building hierarchical_model from pretrained w_model and l_model before finetuning
    
    args.latent_mname = args.waveform_mname+"__"+args.latent_mname
    
    w_ckpt_file = sorted(glob.glob(os.path.join(args.model_dir, args.waveform_mname, "checkpoints", "*.ckpt")))[-1]
    w_yaml_file = os.path.join(args.model_dir, args.waveform_mname, "hparams.yaml")
    l_ckpt_file = sorted(glob.glob(os.path.join(args.model_dir, args.latent_mname, "checkpoints", "*.ckpt")))[-1]
    l_yaml_file = os.path.join(args.model_dir, args.latent_mname, "hparams.yaml")
    
    if args.finetuned:
        args.mname = args.latent_mname+"__finetuned"
        print("\n*** loading finetuned model",args.mname)
        ckpt_file = sorted(glob.glob(os.path.join(args.model_dir,args.mname,"checkpoints","*.ckpt")))[-1]
        yaml_file = os.path.join(args.model_dir,args.mname,"hparams.yaml")
        model = hierarchical_model.load_from_checkpoint(checkpoint_path=ckpt_file,hparams_file=yaml_file,map_location='cpu',
                                                    w_ckpt_file=w_ckpt_file,w_yaml_file=w_yaml_file,l_ckpt_file=l_ckpt_file,l_yaml_file=l_yaml_file)
    else:
        args.mname = args.latent_mname
        print("\n*** loading pretrained waveform and latent models",args.waveform_mname,args.latent_mname)
        model = hierarchical_model(w_ckpt_file=w_ckpt_file,w_yaml_file=w_yaml_file,l_ckpt_file=l_ckpt_file,l_yaml_file=l_yaml_file)
    model.eval()
    
    
    print("\n*** generating grain interpolation of duration",args.interp_len)
    
    audio_interp = make_interp(model,args.interp_len,temperature=args.temperature)
    sf.write(os.path.join(args.export_dir,args.mname+"__sample"+str(args.samples_id)+"_interpolation.wav"),audio_interp,model.w_model.hparams.sr)
    
    
    print("\n*** sampling drums in available classes",model.l_model.hparams.classes)
    
    # here edit manually a chosen drum pattern in 4/4
    
    bpm = 100
    n_bars = 2
    beat_dur = 60/bpm
    bar_dur = 4*beat_dur
    loop_dur = n_bars*bar_dur
    loop_len = int(loop_dur*model.w_model.hparams.sr)
    
    # e.g. classes Clap Cowbell Crash Cymbal Hat Kick Ride Snare Tom
    drum_dict = {"Kick":
                 [[0.,0.6],[0.2,0.4],[0.4,0.6],[0.6,0.4],[0.7,0.2],[0.8,0.6]],
                 "Snare":
                 [           [0.25,0.5],                                   [0.9,0.5]],
                 "Hat":
                 [   [0.1,0.3],   [0.3,0.3], [0.5,0.3],                 [0.85,0.2]],
                 "Clap":
                 [      [0.15,0.2],     [0.45,0.2],            [0.75,0.4]],
                 "Cymb_Crash_Ride":
                 [                                 [0.65,0.6]]}
    # each class is given a list of events [[onset1,amp1],[onset2,amp2],...]
    # onset is in [0,1] as fraction of loop_len
    # amplitude is in [0,1] ~ velocity
    
    output_loop,loop_dict = generate_loop(model,loop_len,drum_dict,temperature=args.temperature)
    sf.write(os.path.join(args.export_dir,args.mname+"__sample"+str(args.samples_id)+"_loop.wav"),output_loop,model.w_model.hparams.sr)
    for cl in drum_dict:
        sf.write(os.path.join(args.export_dir,args.mname+"__sample"+str(args.samples_id)+"_loop_"+cl+".wav"),loop_dict[cl],model.w_model.hparams.sr)
    
    
    
    


