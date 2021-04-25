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


def a5():
    directories.A5()
    # check tensorflow's GPU support
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    features = 784
    classes = 10
    fltr = [3, 5]
    h1 = [128, 384, 749]

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

    for fl in fltr:
        for nodes in h1:
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

            fname = './logs/A5/32({},{})_0.2_{}-10.txt'.format(fl, fl, nodes)
            directories.filecheck(fname)
            acc = []
            vloss = []
            tloss = []
            acc_sum = 0
            loss_sum = 0
            f = open(fname, 'w')
            sys.stdout = f

            fold = 1
            kfold = KFold(5, shuffle=True, random_state=1)
            for train, test in kfold.split(x_train):
                xi_train, xi_test = x_train[train], x_train[test]
                yi_train, yi_test = y_train[train], y_train[test]
                print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')
                history = model.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1, validation_data=(xi_test, yi_test))

                acc.append(history.history['val_accuracy'])
                vloss.append(history.history['val_loss'])
                tloss.append(history.history['loss'])

                # Test the model after training
                test_results = model.evaluate(x_test, y_test, verbose=1)
                print(f'Αποτελέσματα στο fold # {fold} - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
                fold = fold + 1

                # save 5-fold cv results
                loss_sum += test_results[0]
                acc_sum += test_results[1]

            # plots
            # accuracy
            plot_acc = plt.figure(1)
            title1 = 'Validation Accuracy, CNN 32({},{}), max pooling, {}-10'.format(fl, fl, nodes)
            plt.title(title1, loc='center', pad=None)
            plt.plot(np.mean(acc, axis=0))
            plt.ylabel('acc')
            plt.xlabel('epoch')

            # loss
            plot_loss = plt.figure(2)
            title2 = 'Loss, CNN 32({},{}), max pooling, {}-10'.format(fl, fl, nodes)
            plt.title(title2, loc='center', pad=None)
            plt.plot(np.mean(vloss, axis=0))
            # train loss
            plt.plot(np.mean(tloss, axis=0))
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['validation', 'train'], loc='upper left')

            directories.filecheck('./plots/A5/{}.png'.format(title1))
            directories.filecheck('./plots/A5/{}.png'.format(title2))
            plot_acc.savefig('./plots/A5/{}.png'.format(title1), format='png')
            plot_loss.savefig('./plots/A5/{}.png'.format(title2), format='png')

            print(f'Συνολικά αποτελέσματα- Loss {loss_sum/ 5} - Accuracy {acc_sum / 5}')
            # επιστροφή stdout στην κονσόλα
            f.close()
            sys.stdout = sys.__stdout__
            # απελευθερωση μνημης
            print(f'Clearing session....')
            tf.keras.backend.clear_session()
            plt.close(1)
            plt.close(2)
            acc.clear()
            vloss.clear()
            tloss.clear()
