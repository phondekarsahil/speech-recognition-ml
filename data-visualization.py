import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# listing the directory of dataset
print(os.listdir('../speech-recognition'))

# path to audio training set
train_audio_path = '../speech-recognition/train/'

# visualize any one audio signal
samples, sample_rate = librosa.load(train_audio_path+'one/0a7c2a8d_nohash_0.wav', sr = 16000)
fig1 = plt.figure(figsize=(14, 8))
ax1 = fig1.add_subplot(211)
ax1.set_title('Raw wave')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

# as most of the visual related data is at 8000Hz, the new sampling rate is
new_sample_rate = 8000
samples = librosa.resample(samples, sample_rate, new_sample_rate)
ax2 = fig1.add_subplot(212)
ax2.set_title('Resampled wave')
ax2.set_xlabel('time')
ax2.set_ylabel('Amplitude')
ax2.plot(np.linspace(0, new_sample_rate/len(samples), new_sample_rate), samples)

# create a list of available audio labels
labels=os.listdir(train_audio_path)


#  create a list of available audio files for each label
no_of_recordings = []
waves = 0
for label in labels:
    for f in os.listdir(train_audio_path+label):
        if f.endswith('.wav'):
            waves = waves + 1
    no_of_recordings.append(waves)
    waves = 0

# plot a graph of labels vs the no of samples
fig2 = plt.figure(figsize=(30,5))
plt.bar(labels, no_of_recordings)
plt.xlabel('Commmands')
plt.ylabel('No of recordings')
plt.title('No of recordings for each command')

# histograph for duration of recordings

duration_of_recordings = []
for label in labels:
    files = []
    for f in os.listdir(train_audio_path+label):
        if f.endswith('.wav'):
            files.append(f)
    for wav in files:
        sample_rate, samples = wavfile.read(train_audio_path+label+'/'+wav)
        duration_of_recordings.append(float(len(samples))/sample_rate)
    
plt.hist(duration_of_recordings)
plt.show()