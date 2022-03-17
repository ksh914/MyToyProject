import tensorflow as tf
import torch as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
wav_file = '/Users/kim/Desktop/github/sample_sound/p232_001.wav'

#1
y,sr = librosa.load(wav_file,sr=16000)
S = np.abs(librosa.stft(y))
fig,ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),
                             y_axis='log',x_axis='time',ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img,ax=ax,format="%+2.0f dB")

#2
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
fig,ax = plt.subplots()
S_dB = librosa.power_to_db(S,ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                              y_axis='mel', sr=sr,
                              fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')

#3
ML = []
wav_file = '/Users/kim/Desktop/github/sample_sound/'
for i in wav_file:
    y,sr = librosa.load(os.path.join(wav_file),sr=16000)
    ML.append(y)
magnitude=np.abs(y)
lob_spectrogram = librosa.amplitude_to_db(magnitude)
plt.figure(figszie=(10,4))
librosa.display.specshow(log_spectrogram,sr=16000,hop_length=64)
plt.xlabel('Time')
plt.ylabel('Frequency')
