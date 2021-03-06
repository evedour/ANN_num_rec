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
from sklearn import preprocessing


def single_layer(ep):
    # αρχικοποίηση directories αποθήκευσης
    directories.single_layer()
    # GPU support
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print("CUDA - Αριθμός διαθέσιμων GPUs:", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    # αρχικοποίηση μεταβλητών
    features = 784
    classes = 10
    h1 = [10, 397, 794]

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

    # έλεγχος για όλα τα H1
    for h_1 in h1:
        # αρχικοποίηση sum μετρήσεων
        loss_sum = 0
        acc_sum = 0
        # ορισμός directory αποθήκευσης stdout
        f_ce = "./logs/A2/Single_Layer/results_CE_%s.txt" % h_1
        f_mse = "./logs/A2/Single_Layer/results_MSE_%s.txt" % h_1
        directories.filecheck(f_ce)
        directories.filecheck(f_mse)

        # δημιουργία μοντέλων με χρήση του keras API
        model_ce = Sequential()
        model_mse = Sequential()
        # πρώτο κρυφό επίπεδο
        model_ce.add(Dense(h_1, input_shape=input_shape, activation='relu'))
        model_mse.add(Dense(h_1, input_shape=input_shape, activation='relu'))
        # επίπεδο εξόδου
        model_ce.add(Dense(classes, activation='softmax'))
        model_mse.add(Dense(classes, activation='softmax'))

        # compile
        # crossentropy
        model_ce.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        # mse
        model_mse.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
        # άνοιγμα αρχείου εξόδου
        f = open(f_ce, 'w')
        print('Μοντέλο Cross Entropy Loss για {} κόμβους στο πρώτο κρυφό επίπεδο'.format(h_1))
        sys.stdout = f
        ################################################################################################################
        ###################################### CROSS ENTROPY 5-FOLD CV #################################################
        aval = []
        lval = []
        ltrain = []
        fold = 1
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            # διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            # fit μοντέλου
            ce_history = model_ce.fit(xi_train, yi_train, epochs=ep, batch_size=200, verbose=1,
                                      validation_data=(xi_test, yi_test))
            # αποθήκευση validation metrics για τα plots
            aval.append(ce_history.history['val_accuracy'])
            lval.append(ce_history.history['val_loss'])
            ltrain.append(ce_history.history['loss'])

            # μετρήσεις μοντέλου
            ce_results = model_ce.evaluate(x_test, y_test, verbose=1)
            print(f'Αποτελέσματα στο fold # {fold} - Loss: {ce_results[0]} - Accuracy: {ce_results[1]}')

            fold = fold + 1
            # αποθήκευση για προβολή των αποτελεσμάτων 5-fold CV
            loss_sum += ce_results[0]
            acc_sum += ce_results[1]

        # plots
        # accuracy
        plot_acc = plt.figure(1)
        title1 = 'Validation Accuracy Cross Entropy Model {}-10'.format(h_1)
        plt.title(title1, loc='center', pad=None)
        plt.plot(np.mean(aval, axis=0))
        plt.ylabel('acc')
        plt.xlabel('epoch')

        # loss
        plot_loss = plt.figure(2)
        title2 = 'Loss Cross Entropy Model {}-10'.format(h_1)
        plt.title(title2, loc='center', pad=None)
        # validation loss
        plt.plot(np.mean(lval, axis=0))
        # train loss
        plt.plot(np.mean(ltrain, axis=0))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['validation', 'train'], loc='upper left')

        # Save locally
        directories.filecheck('./plots/A2/Single_Layer/{}.png'.format(title1))
        directories.filecheck('./plots/A2/Single_Layer/{}.png'.format(title2))
        plot_acc.savefig('./plots/A2/Single_Layer/{}.png'.format(title1), format='png')
        plot_loss.savefig('./plots/A2/Single_Layer/{}.png'.format(title2), format='png')

        # εκτύπωση αποτελεσμάτων
        print(f'Συνολικά Αποτελέσματα (Cross Entropy Model)- Loss {loss_sum/5} - Accuracy {acc_sum/5}')
        # επιστροφή stdout στην κονσόλα
        f.close()
        sys.stdout = sys.__stdout__
        # απελευθέρωση μνήμης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)

        # αρχικοποίηση καινούριων μεταβλητών
        loss_sum = 0
        acc_sum = 0
        fold = 1
        aval.clear()
        lval.clear()
        ltrain.clear()
        # νεο αρχείο εξόδου
        f = open(f_mse, 'w')
        print('Μοντέλο Mean Squared Error Loss για {} κόμβους στο πρώτο κρυφό επίπεδο'.format(h_1))
        sys.stdout = f
        ################################################################################################################
        #################################### MEAN SQUARED ERROR 5-FOLD CV ##############################################
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            # διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            # fit μοντέλου
            mse_history = model_mse.fit(xi_train, yi_train, epochs=ep, batch_size=200, verbose=1,
                                        validation_data=(xi_test, yi_test))
            # αποθήκευση validation metrics για τα plots
            aval.append(mse_history.history['val_accuracy'])
            lval.append(mse_history.history['val_loss'])
            ltrain.append(mse_history.history['loss'])

            # μετρήσεις μοντέλου
            mse_results = model_mse.evaluate(x_test, y_test, verbose=1)
            print(f'Αποτελέσματα στο fold # {fold} - Loss: {mse_results[0]} - Accuracy: {mse_results[1]}')

            fold = fold + 1
            # αποθήκευση για προβολή των αποτελεσμάτων 5-fold CV
            loss_sum += mse_results[0]
            acc_sum += mse_results[1]

        # plots
        # accuracy
        plot_acc = plt.figure(1)
        title1 = 'Validation Accuracy MSE Model {}-10'.format(h_1)
        plt.title(title1, loc='center', pad=None)
        plt.plot(np.mean(aval, axis=0))
        plt.ylabel('acc')
        plt.xlabel('epoch')

        # loss
        plot_loss = plt.figure(2)
        title2 = 'Loss MSE Model {}-10'.format(h_1)
        plt.title(title2, loc='center', pad=None)
        # validation loss
        plt.plot(np.mean(lval, axis=0))
        # train loss
        plt.plot(np.mean(ltrain, axis=0))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['validation', 'train'], loc='upper left')

        # Save locally
        directories.filecheck('./plots/A2/Single_Layer/{}.png'.format(title1))
        directories.filecheck('./plots/A2/Single_Layer/{}.png'.format(title2))
        plot_acc.savefig('./plots/A2/Single_Layer/{}.png'.format(title1), format='png')
        plot_loss.savefig('./plots/A2/Single_Layer/{}.png'.format(title2), format='png')

        # εκτύπωση αποτελεσμάτων
        print(f'Results sum (MSE) - Loss {loss_sum/5} - Accuracy {acc_sum/5}')
        f.close()
        sys.stdout = sys.__stdout__
        # καθαρισμός μνήμης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)
