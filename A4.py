import threading

import directories
import tensorflow
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
from sklearn import preprocessing


def a4():
    # αρχικοποίηση directories αποθήκευσης
    directories.A4()

    # GPU support
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print("CUDA - Αριθμός διαθέσιμων GPUs:", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    # αρχικοποίηση μεταβλητών
    features = 784
    classes = 10
    loss_fun = 'categorical_crossentropy'
    h1 = 794
    h2 = 100
    learning_rate = 0.1
    m = 0.6
    reg_factors = [0.1, 0.5, 0.9]

    # φόρτωση mnist από το keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # κάνουμε το mnist reshape
    x_train = x_train.reshape(x_train.shape[0], features)
    x_test = x_test.reshape(x_test.shape[0], features)

    # MinMax scaling
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # ορισμός των labels
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # ορισμός input shape για το μοντέλο MLP βάσει των χαρακτηριστικών
    input_shape = (features,)
    opt = tensorflow.keras.optimizers.SGD(lr=learning_rate, momentum=m, decay=0.0, nesterov=False)

    for factor in reg_factors:
        # Δημιουργία μοντέλου με χρήση του keras API
        model = Sequential()
        # Πρώτο κρυφό επίπεδο
        model.add(Dense(h1, kernel_regularizer=l2(factor), bias_regularizer=l2(factor), input_shape=input_shape,
                        activation='relu'))
        # Δεύτερο κρυφό επίπεδο
        model.add(Dense(h2, activation='relu'))
        # Επίπεδο εξόδου
        model.add(Dense(classes, activation='softmax'))

        fname = "./logs/A4/results_{}.txt".format(factor)
        directories.filecheck(fname)

        # compile
        model.compile(loss=loss_fun, optimizer=opt, metrics=['accuracy'])

        fold = 1
        loss_sum = 0
        acc_sum = 0
        aval = []
        lval = []
        ltrain = []
        f = open(fname, 'w')
        sys.stdout = f

        # 5-fold CV
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            # διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            # fit μοντέλου
            history = model.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1,
                                validation_data=(xi_test, yi_test),
                                callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])

            # στατιστικά
            aval.append(np.mean(history.history['val_accuracy']))
            lval.append(np.mean(history.history['val_loss']))
            ltrain.append(np.mean(history.history['loss']))

            # μετρησεις μοντέλου
            results = model.evaluate(x_test, y_test, verbose=1)
            print(f'Αποτελέσματα στο fold # {fold} - Loss: {results[0]} - Accuracy: {results[1]}')

            fold += fold
            # αποθήκευση για προβολή των αποτελεσμάτων 5-fold CV
            loss_sum += results[0]
            acc_sum += results[1]

        # plots
        # accuracy
        plot_acc = plt.figure(1)
        title1 = 'Validation Accuracy r='.format(factor)
        plt.title(title1, loc='center', pad=None)
        plt.plot(aval)
        plt.ylabel('acc')
        plt.xlabel('epoch')

        # loss
        plot_loss = plt.figure(2)
        title2 = 'Loss r={}'.format(factor)
        plt.title(title2, loc='center', pad=None)
        # validation loss
        plt.plot(lval)
        # train loss
        plt.plot(ltrain)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['validation', 'train'], loc='upper left')

        # Save locally
        directories.filecheck('./plots/A4/{}.png'.format(title1))
        directories.filecheck('./plots/A4/{}.png'.format(title2))
        plot_loss.savefig("./plots/A4/{}.png".format(title2), format='png')
        plot_acc.savefig("./plots/A4/{}.png".format(title1), format='png')

        # εκτύπωση αποτελεσμάτων
        print(f'Συνολικά Αποτελέσματα - Loss {loss_sum / 5} - Accuracy {acc_sum / 5}')
        # επιστροφή stdout στην κονσόλα
        f.close()
        sys.stdout = sys.__stdout__
        # απελευθερωση μνημης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)
        aval.clear()
        lval.clear()
        ltrain.clear()
