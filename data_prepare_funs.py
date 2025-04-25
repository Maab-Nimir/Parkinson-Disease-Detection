import os
import json
import torchaudio

from speechbrain.utils.data_utils import get_all_files
import pandas as pd
from sklearn.model_selection import train_test_split

import json
from collections import Counter


def extract_metadata(data_files):
    data = []
    
    for wav_path in data_files:
        parts = wav_path.split(os.sep)
    
        # Assumes folder format: /home/ulaval.ca/maelr5/scratch/parkinsons/<label-folder>/<speaker>/<file.wav>
        speaker_id = parts[-2]    # e.g., "Davide S"
        filename = parts[-1]
        
        if "Healthy Control" in wav_path:
            label_folder = parts[-3]  # e.g: "15 Young Healthy Control"
        elif "with Parkinson's disease" in wav_path:
            label_folder = parts[-4] # e.g: "28 People with Parkinson's disease"
    
        # Determine label from folder name
        label = "HC" if "Healthy Control" in label_folder else "PD"
    
        data.append({
            "filename": filename,
            "full_path": wav_path,
            "speaker_id": speaker_id,
            "label": label
        })
    
    df = pd.DataFrame(data)
    return df

def split_by_speaker(dataframe, seed_value = 1986):
    speakers = dataframe['speaker_id'].unique()
    print('data speakers size= ', len(speakers))
    train_speakers, eval_speakers = train_test_split(speakers, test_size=0.2, train_size=0.8, random_state=seed_value, shuffle=True)
    valid_speakers, test_speakers = train_test_split(eval_speakers, test_size=0.5, train_size=0.5, random_state=seed_value, shuffle=True)
    print('train speakers size= ', len(train_speakers))
    print('valid speakers size= ', len(valid_speakers))
    print('test speakers size= ', len(test_speakers))
    
    train_df = dataframe[dataframe['speaker_id'].isin(train_speakers)]
    valid_df = dataframe[dataframe['speaker_id'].isin(valid_speakers)]
    test_df = dataframe[dataframe['speaker_id'].isin(test_speakers)]
    
    return train_df, valid_df, test_df

def df_to_json(df, json_path, shuffle=True, balance_classes=False, seed=42):
    if balance_classes:
        # Split into PD and HC
        pd_df = df[df['label'] == 'PD']
        hc_df = df[df['label'] == 'HC']

        # Find the smaller class size
        min_size = min(len(pd_df), len(hc_df))

        # Downsample both classes
        pd_df = pd_df.sample(n=min_size, random_state=seed)
        hc_df = hc_df.sample(n=min_size, random_state=seed)

        # Combine and shuffle
        df = pd.concat([pd_df, hc_df], axis=0)
    
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    data = {}
    for _, row in df.iterrows():
        utt_id = os.path.splitext(row['filename'])[0]  # unique id
        # Getting info
        audioinfo = torchaudio.info(row['full_path'])
        # Compute the duration in seconds.
        # This is the number of samples divided by the sampling frequency
        duration = audioinfo.num_frames / audioinfo.sample_rate
        
        data[utt_id] = {
            "path": row['full_path'],
            "spk_id": row['speaker_id'],
            "length": duration,
            "detection": row['label']
        }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def load_paths(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return set(ex["path"] for ex in data.values())

def load_speakers(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return set(ex["spk_id"] for ex in data.values())

def calculate_class_samples(json_path):
    
    # Load the JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Count how many samples belong to each detection class
    class_counts = Counter()
    
    for sample in data.values():
        detection_class = sample["detection"]
        class_counts[detection_class] += 1
    
    # Print the result
    for cls, count in class_counts.items():
        print(f"{cls}: {count} samples")


def df_to_small_json(df, json_path, shuffle=True, balance_classes=False, seed=42, samples_per_class=None):
    if balance_classes:
        # Split into PD and HC
        pd_df = df[df['label'] == 'PD']
        hc_df = df[df['label'] == 'HC']

        # Optionally limit samples per class
        if samples_per_class:
            pd_df = pd_df.sample(n=min(samples_per_class, len(pd_df)), random_state=seed)
            hc_df = hc_df.sample(n=min(samples_per_class, len(hc_df)), random_state=seed)
        
        df = pd.concat([pd_df, hc_df], axis=0)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    data = {}
    for _, row in df.iterrows():
        utt_id = os.path.splitext(os.path.basename(row['filename']))[0]
        audioinfo = torchaudio.info(row['full_path'])
        duration = audioinfo.num_frames / audioinfo.sample_rate

        data[utt_id] = {
            "path": row['full_path'],
            "spk_id": row['speaker_id'],
            "length": duration,
            "detection": row['label']
        }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)



