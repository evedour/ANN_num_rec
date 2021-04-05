import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

#x is the data
#y is the label
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(y_train.shape)
numclass = 10
Y_train=tf.keras.utils.to_categorical(y_train, numclass)
Y_test = tf.keras.utils.to_categorical(y_test, numclass)
#5-fold cv
kfold = KFold(5, shuffle=True, random_state=1)
for train, test in kfold.split(x_train):
    print("TRAIN:", train, "TEST:", test)
    X_train, X_test = x_train[train], x_train[test]
    # reshape
    X_train = X_train.reshape(48000, 784)
    X_test = X_test.reshape(12000, 28,28)
    X_test = X_test.reshape(12000,784)
    #normalize
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    #TODO softmax



