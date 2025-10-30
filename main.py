from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import librosa
#import soundfile as sf

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

class UrbanSoundDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_info = self.annotations.iloc[idx]
        file_name = audio_info["slice_file_name"]
        fold = f"fold{audio_info['fold']}"
        label = audio_info["classID"]

        file_path = os.path.join(self.audio_dir, fold, file_name)

        signal, sample_rate = librosa.load(file_path, sr=self.target_sample_rate)
        signal = torch.from_numpy(signal.astype(np.float32))

        signal = self.transformation(signal)

        return signal, label



if __name__ == "__main__":
    csv_file = "input/urbansound8k/UrbanSound8K.csv"
    audio_dir = "input/urbansound8k/"
    sample_rate = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)

    USDataset = UrbanSoundDataset(csv_file, audio_dir, mel_spectrogram, sample_rate)

    wave, label = USDataset[0]
