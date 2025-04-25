#!/bin/bash
module load python/3.10
source parksenv/bin/activate

# Hubert
# python ssl/trainwav2vec.py ssl/hparams_hubert_phrases.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=10 --use_tensorboard=True


# ECAPATDNN from scratch
rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/fromscratch/full_dataset/fbank/1986
python train_models/train_ecapatdnn_fbanks.py train_models/hparams_fromscratch_ecapatdnn_fbanks.yaml  --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True


# ECAPATDNN + MLP Ensemble
# Delete the output folder to start training from scratch (and not from a previous checkpoint).
# rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ecapamlpensembleavg/full_dataset/fbank/1986
# python train_models/train_ecapamlpensembleavg_fbanks.py train_models/hparams_ecapamlpensembleavg_fbanks.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True


# rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/fromscratch/augmentation/full_dataset/fbank/1986
# python train_models/train_ecapatdnnaugmented_fbanks.py train_models/hparams_fromscratch_ecapatdnnaugmented_fbanks.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True


