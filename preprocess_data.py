import os
import pandas as pd
import numpy as np
import asyncio
from pydub import AudioSegment
import librosa
from io import BytesIO

import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectrogram = self.compute_spectrogram(row['audio_path'])
        return {'spectrogram': spectrogram, 'label': row['gender']}

    def compute_spectrogram(self, audio_path):
        audio = AudioSegment.from_file(audio_path)
        wav_file = BytesIO()
        audio.export(wav_file, format='wav')
        wav_file.seek(0)
        with wav_file as file:
            y, sr = librosa.load(file, sr=None, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB


# Load the TSV files
validated_df = pd.read_csv('cv_corpus/en/validated.tsv', sep='\t')
invalidated_df = pd.read_csv('cv_corpus/en/invalidated.tsv', sep='\t')
other_df = pd.read_csv('cv_corpus/en/other.tsv', sep='\t')

# Concatenate the dataframes into one
df = pd.concat([validated_df, invalidated_df, other_df])

# Define the paths to the audio files
audio_dir = 'cv_corpus/en/clips'
df['audio_path'] = df['path'].apply(lambda x: os.path.join(audio_dir, x))

print('Working on adding audio files to df...')

# Split the dataframe into batches
batch_size = 1000
batch_dfs = np.array_split(df[:10], len(df[:10]) // batch_size + 1)

# Define a function to process a batch of audio files
async def process_batch(batch_df, statement):
    print(statement)
    dataset = SpectrogramDataset(batch_df)
    spectrograms = []
    for i in range(len(dataset)):
        spectrogram = dataset[i]['spectrogram']
        spectrograms.append(spectrogram)
    batch_df['spectrogram'] = spectrograms
    return batch_df

# Process the batches asynchronously
async def main():
    tasks = []
    for i, batch_df in enumerate(batch_dfs):
        statement = f"Processing batch {i+1} of {len(batch_dfs)}"
        task = asyncio.create_task(process_batch(batch_df, statement))
        tasks.append(task)
    result_dfs = await asyncio.gather(*tasks)
    final_df = pd.concat(result_dfs)
    return final_df

final_df = asyncio.run(main())
final_df.to_json('data.json')
