import os
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from keras.models import load_model

train_audio_path = '../speech-recognition/train/'

all_waves = []
all_labels = []

labels=os.listdir(train_audio_path)

labels = ['zero','one','two','three','four','five','six','seven','eight','nine']
#labels = ['zero','one','two','three']
#labels = ['zero','one']

for label in labels:
    print(label)
    waves = []
    for f in os.listdir(train_audio_path+label):
        if f.endswith('.wav'):
            waves.append(f)
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path+label+'/'+wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if len(samples) == 8000:
            all_waves.append(samples)
            all_labels.append(label)
            
le = LabelEncoder()
Y = le.fit_transform(all_labels)
classes = list(le.classes_)

Y = np_utils.to_categorical(Y, num_classes = len(labels))

all_waves = np.array(all_waves).reshape(-1,8000,1)

x_train, x_val, y_train, y_val = train_test_split(np.array(all_waves), np.array(Y), 
                                                  stratify=Y, test_size = 0.2, random_state = 777, 
                                                  shuffle = True)

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000, 1))

# First conv1D layer
conv = Conv1D(8, 13,  padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Second conv1D layer
conv = Conv1D(16, 11,  padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Third conv1D layer
conv = Conv1D(32, 9,  padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
            
# Fourth conv1D layer
conv = Conv1D(64, 7,  padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Flatten layer
conv = Flatten()(conv)

# Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

# Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model/best_model.h5', monitor='val_acc', verbose=1, save_best_only=True,mode='max')

history = model.fit(x_train, y_train, epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val, y_val))

model.save('model.h5')
print('model saved')

#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
#
model = load_model('model.h5')

def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


import random
index = random.randint(0, len(x_val)-1)
samples = x_val[index].ravel()
print('Audio: ',classes[np.argmax(y_val[index])])
print('Text: ', predict(samples))

mysamples, mysample_rate = librosa.load('../speech-recognition/two.wav', sr = 16000)
mysamples = librosa.resample(mysamples, mysample_rate, 8000)
myprediction = predict(mysamples)
print(myprediction)
os.listdir()