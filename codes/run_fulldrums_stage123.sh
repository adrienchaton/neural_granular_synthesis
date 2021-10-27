# training stages 1, 2 and 3 on the drum dataset
# requires a 12GB GPU

CUDA_VISIBLE_DEVICES=0 python train_stage1.py --mname fulldrums_22k_L2048_Rej0202_normola --batch_size 24 --max_steps 300000 --classes Clap Cowbell Cymb_Crash_Ride Hat Kick Snare Tom --tar_beta 0.003 --w_config ./configs/w_22k_L2048_Reject0202_normola.json --data_dir /fast-2/adrien/NGSS/drum_data/full_dataset/

CUDA_VISIBLE_DEVICES=0 python train_stage2.py --mname embedding_cond_l_E256_1LSTM --waveform_mname fulldrums_22k_L2048_Rej0202_normola --conditional --batch_size 32 --max_steps 500000 --tar_beta 0.01 --l_config ./configs/l_E256_1LSTM.json

CUDA_VISIBLE_DEVICES=0 python train_stage3.py --latent_mname embedding_cond_l_E256_1LSTM --waveform_mname fulldrums_22k_L2048_Rej0202_normola --batch_size 24 --max_steps 200000

