import tensorflow as tf
import torch as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

x=librosa.load('/Users/kim/Desktop/github/sample_sound/p232_001.wav',sr=16000)[0] # librosa.load(location, sr=22050)
y=librosa.stft(x,n_fft=128,hop_length=64,win_length=128) #stft : Short-Time Fourier Transform : librosa.stft()

magnitude=np.abs(y) #y의 절댓값 계산
log_spectrogram = librosa.amplitude_to_db(magnitude)

plt.figure(figsize=(10,4))
librosa.display.specshow(log_spectrogram,sr=16000,hop_length=64)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title("spectrogram(dB")