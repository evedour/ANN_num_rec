import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import numpy as np
import directories
import sys


features = 784
classes = 10
fl = 5
nodes = 384

# φόρτωση mnist από το keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# κάνουμε το mnist reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ορισμός των labels
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)


print(f'Μοντέλο CNN με ένα επίπεδο 32 φίλτρων {fl}x{fl}, 2x2 MaxPooling και MLP {nodes}:10')
# model
model = Sequential()
# first layer: Convolution 2D, 32 filters of size 5x5
model.add(Conv2D(32, (fl, fl), input_shape=(28, 28, 1), activation='relu', kernel_initializer='he_uniform'))
# second layer: MaxPooling 2D, returns max value of image portion (2x2)
model.add(MaxPooling2D(pool_size=(2, 2)))
# third layer: Flatten results of previous layers to feed into the MLP
model.add(Flatten())
# fourth and output layer: our standard MLP
model.add(Dense(nodes, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='he_uniform', activation='softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

acc = []
vloss = []
tloss = []
acc_sum = 0
loss_sum = 0


fold = 1
kfold = KFold(5, shuffle=True, random_state=1)
for train, test in kfold.split(x_train):
    xi_train, xi_test = x_train[train], x_train[test]
    yi_train, yi_test = y_train[train], y_train[test]
    print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')
    history = model.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1, validation_data=(xi_test, yi_test))

    # Test the model after training
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Αποτελέσματα στο fold # {fold} - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
    fold = fold + 1

    # save 5-fold cv results
    loss_sum += test_results[0]
    acc_sum += test_results[1]

print(f'Συνολικά αποτελέσματα- Loss {loss_sum/ 5} - Accuracy {acc_sum / 5}')
model.save_weights('./my_model_weights.h5')
