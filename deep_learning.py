import os
import keras
import numpy as np
from sklearn import decomposition
from utils import *

# ========================== fine tunning ==============================
# optimizer     learning rate   epochs  batch size  test loss   test acc
#  Adam             0.001           [loss stays unchange]
#  SGD              0.001         16        64        0.498       0.710
#  SGD              0.001         16        128       0.600       0.698 
#  SGD              0.001         32        128       0.467       0.768
#  SGD              0.001         64        128       0.360       0.819
#  SGD              0.001        128        128       0.412       0.769
#  SGD             0.0001        128        128       0.362       0.819
# AdaGrad           0.01         128        128       7.312       0.547
# RMSprop           0.001         30         32        1.92       0.873  [Diemsion Reduction + 1 more layer]
# ======================================================================

def load_data(dir=DATA_PATH):
    data = np.load(dir+'data.npy')
    label = np.load(dir+'target.npy')
    print("Successfully load data and labels.")
    return data, label

# Load data....
data, label = load_data()
label_ = keras.utils.to_categorical(label)

# Dimension Reduction
pca = decomposition.PCA(n_components=100)
pca.fit(data)
data = pca.transform(data)
print(data.shape)
N = int(len(data)*0.8)
print("Dimension Reduction complete")

train_X = keras.utils.normalize(data[:N])
test_X = keras.utils.normalize(data[N:])
# train_X = data[:N]
# test_X = data[N:]
train_Y = label_[:N]
test_Y = label_[N:]

data_num, features =  data.shape
model = keras.models.Sequential()
model.add(keras.layers.Dense(1024, input_shape=(features, ),
                                  activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.RMSprop(),
              metrics=["accuracy"])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model.fit(train_X, train_Y,
          batch_size=32,
          epochs=30,
          verbose=1)

model.save('deep_model/deep.h5')

test_loss, test_acc = model.evaluate(test_X, test_Y)
print("test loss:%f\ttest acc:%f" % (test_loss, test_acc))

