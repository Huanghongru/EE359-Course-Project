import os
import keras
import numpy as np
from keras import layers
from utils import *

def load_data(dir=DATA_PATH):
    data = np.load('data.npy')
    label = np.load('target.npy')
    print("Successfully load data and labels.")
    return data, label

data, label = load_data()
label_ = keras.utils.to_categorical(label)
data_num, features =  data.shape
model = keras.models.Sequential()
model.add(keras.layers.Dense(1024, input_shape=(features, ),
                                  activation="relu"))
model.add(keras.layers.Dense(528, activation="relu"))
model.add(keras.layers.Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(data, label_,
          batch_size=64,
          epochs=2,
          validation_split=0.1,
          verbose=1)

model.save('deep_model/deep.h5')
