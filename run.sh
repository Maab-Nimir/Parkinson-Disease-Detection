#!/bin/bash
module load python/3.10
source parksenv/bin/activate

# rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/train_with_wav2vec2/base/1986
# python ssl/trainwav2vec.py ssl/hparams_wav2vec.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=5

rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/full_dataset/fbank/1986
python train_models/train_ecapatdnn_fbanks.py train_models/hparams_ecapatdnn_fbanks.yaml  --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0'