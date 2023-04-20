import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import asyncio

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

# Define a function to process a batch of audio files
async def process_batch(batch_df, statement):
    print(statement)
    spectrograms = []
    for audio_path in batch_df['audio_path']:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        spectrograms.append(S_dB)
    batch_df['spectrogram'] = spectrograms
    return batch_df

# Split the dataframe into batches
batch_size = 1000
batch_dfs = np.array_split(df, len(df) // batch_size + 1)

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
