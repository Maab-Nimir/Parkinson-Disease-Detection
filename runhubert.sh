#!/bin/bash
module load python/3.10
source parksenv/bin/activate

python ssl/trainwav2vec.py ssl/hparams_hubert.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=10 --use_tensorboard=True

# rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/full_dataset/fbank/1986
# python train_models/train_ecapatdnn_fbanks.py train_models/hparams_ecapatdnn_fbanks.yaml  --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True
