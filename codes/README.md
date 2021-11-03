# Pytorch-Lightning codes for training a hierarchical model conditionned on drum classes.

It is directly handling specific training features such as half-precision (not tested) and logging/profiling.

## Install dependencies

python3.7 -m venv venv_tmp

source venv_tmp/bin/activate

pip install -r requirements.txt

## Train own model

### Configure models and training hyper-parameters

Use default model configurations in ./configs/ or create custom ones (cf. commented parts at the end of models.py)

Use default argument parsers or create custom ones in run_fulldrums_stage123.sh

### Prepare dataset

--data_dir argument of train_stage1.py should contain folders with .wav files with start of audio already trimmed (onset time)

each folder refers to an instrument class on which the decoder model is conditionned for sampling

if using other drum classes, update --classes argument of train_stage1.py to correspond to these folder names

we provide checkpoints for a model pretrained on a dataset of sire #train = 13264 and #test = 2346 with the following classes

Clap, Cowbell, Cymb_Crash_Ride, Hat, Kick, Snare, Tom

### Training

bash run_fulldrums_stage123.sh which runs on a 12GB NVIDIA GPU

stage 1 = pretraining the waveform model for auto-encoding individual audio grains

stage 2 = pretraining the embedding model for auto-encoding series of latent grain features of fixed duration (--tar_l argument of train_stage1.py)

stage 3 = optional fine-tuning by end-to-end backpropagation of all losses through both VAEs

## Generate

Either load a fine-tuned hierarchical model or the pretraining checkpoints in fun.py

Generate variable-length interpolations in the grain latent space

Generate a loop by sampling the embedding with conditioning on the target drum class

note: the fine-tuned checkpoint is not uploaded due to upload size limits, feel free to ask for it if needed
