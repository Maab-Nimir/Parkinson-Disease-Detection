#!/bin/bash
module load python/3.10
source parksenv/bin/activate

# python ssl/trainwav2vec.py ssl/hparams_wav2vec.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=10 --use_tensorboard=True

# python ssl/trainwav2vec.py ssl/hparams_wavlm.yaml --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=50 --use_tensorboard=True

# rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/full_dataset/fbank/1986
# python train_models/train_ecapatdnn_fbanks.py train_models/hparams_ecapatdnn_fbanks.yaml  --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True


rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/fromscratch/phrases/fbank/1986
# Run Training
python train_models/train_ecapatdnn_fbanks.py train_models/hparams_fromscratch_phrases_ecapatdnn_fbanks.yaml  --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True



# rm -rf /home/ulaval.ca/maelr5/scratch/parkinsons-results/ECAPA-TDNN/wav2vecfs/full_dataset/1986
# python train_models/train_wav2vec_ecapatdnn.py train_models/hparams_wav2vec_ecapatdnn.yaml  --data_folder='/home/ulaval.ca/maelr5/scratch/parkinsons' --device='cuda:0' --number_of_epochs=15 --use_tensorboard=True