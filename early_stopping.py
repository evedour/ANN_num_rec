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


def early_stopping():
    # αρχικοποίηση directories αποθήκευσης
    directories.early_stopping()

    # GPU support
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print("CUDA - Αριθμός διαθέσιμων GPUs:", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    # αρχικοποίηση μεταβλητών
    features = 784
    classes = 10
    h1 = 794
    h2 = 200

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

    loss_sum = 0
    acc_sum = 0
    f_ce = "./logs/A2/Early_Stopping/results_CE_%s-%s.txt" % (h1, h2)
    f_mse = "./logs/A2/Early_Stopping/results_MSE_%s-%s.txt" % (h1, h2)
    directories.filecheck(f_ce)
    directories.filecheck(f_mse)

    # Δημιουργία μοντέλων με χρήση του keras API
    model_ce = Sequential()
    model_mse = Sequential()
    # πρώτο κρυφό επίπεδο
    model_ce.add(Dense(h1, input_shape=input_shape, activation='relu'))
    model_ce.add(Dense(h2, activation='relu'))
    # δεύτερο κρυφό επίπεδο
    model_mse.add(Dense(h1, input_shape=input_shape, activation='relu'))
    model_mse.add(Dense(h2, activation='relu'))
    # επίπεδο εξόδου
    model_ce.add(Dense(classes, activation='softmax'))
    model_mse.add(Dense(classes, activation='softmax'))

    # compile
    # crossentropy
    model_ce.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    # mse
    model_mse.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])

    # αρχείο εξόδου
    f = open(f_ce, 'w')
    print('Μοντέλο Cross Entropy Loss για {} κόμβους στο πρώτο κρυφό επίπεδο και {} στο δεύτερο, με early stopping'.format(h1, h2))
    sys.stdout = f
    ################################################################################################################
    ###################################### CROSS ENTROPY 5-FOLD CV #################################################
    fold = 1
    kfold = KFold(5, shuffle=True, random_state=1)
    aval = []
    lval = []
    ltrain = []
    for train, test in kfold.split(x_train):
        # διαχωρισμός train-test indexes
        xi_train, xi_test = x_train[train], x_train[test]
        yi_train, yi_test = y_train[train], y_train[test]
        print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

        # fit μοντέλου
        ce_history = model_ce.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1,
                                  validation_data=(xi_test, yi_test),
                                  callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])

        # στατιστικά
        aval.append(np.mean(ce_history.history['val_accuracy']))
        lval.append(np.mean(ce_history.history['val_loss']))
        ltrain.append(np.mean(ce_history.history['loss']))

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
    title1 = 'Validation Accuracy Crossentropy Model with Early stopping {}-{}-10'.format(h1, h2)
    plt.title(title1, loc='center', pad=None)
    plt.plot(aval)
    plt.ylabel('acc')
    plt.xlabel('epoch')
    directories.filecheck('./plots/A2/Early_Stopping/{}.png'.format(title1))
    plot_acc.savefig("./plots/A2/Early_Stopping/{}.png".format(title1), format='png')

    # loss
    plot_loss = plt.figure(2)
    title2 = 'Loss Crossentropy Model with Early Stopping {}-{}-10'.format(h1, h2)
    plt.title(title2, loc='center', pad=None)
    # validation loss
    plt.plot(lval)
    # train loss
    plt.plot(ltrain)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'train'], loc='upper left')

    # Save locally
    directories.filecheck('./plots/A2/Early_Stopping/{}.png'.format(title2))
    plot_loss.savefig("./plots/A2/Early_Stopping/{}.png".format(title2), format='png')


    # εκτυπωση αποτελεσμάτων
    print(f'Συνολικά Αποτελέσματα (Cross Entropy Model)- Loss {loss_sum / 5} - Accuracy {acc_sum / 5}')
    # επιστροφή stdout στην κονσόλα
    f.close()
    sys.stdout = sys.__stdout__
    # απελευθερωση μνημης
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
    print('Μοντέλο Mean Squared Error Loss για {} κόμβους στο πρώτο κρυφό επίπεδο και {} στο δεύτερο, με early stopping'.format(h1, h2))
    sys.stdout = f
    ####################################################################################################################
    #################################### MEAN SQUARED ERROR 5-FOLD CV ##################################################
    kfold = KFold(5, shuffle=True, random_state=1)
    for train, test in kfold.split(x_train):
        # διαχωρισμός train-test indexes
        xi_train, xi_test = x_train[train], x_train[test]
        yi_train, yi_test = y_train[train], y_train[test]
        print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

        # fit μοντέλου
        mse_history = model_mse.fit(xi_train, yi_train, epochs=50, batch_size=200, verbose=1,
                                    validation_data=(xi_test, yi_test),
                                    callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])

        # αποθήκευση validation metrics για τα plots
        aval.append(np.mean(mse_history.history['val_accuracy']))
        lval.append(np.mean(mse_history.history['val_loss']))
        ltrain.append(np.mean(mse_history.history['loss']))

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
    title1 = 'Validation Accuracy MSE Model {}-{}-10'.format(h1, h2)
    plt.title(title1, loc='center', pad=None)
    plt.plot(aval)
    plt.ylabel('acc')
    plt.xlabel('epoch')

    # loss
    plot_loss = plt.figure(2)
    title2 = 'Loss MSE Model {}-{}-10'.format(h1, h2)
    plt.title(title2, loc='center', pad=None)
    # validation loss
    plt.plot(lval)
    # train loss
    plt.plot(ltrain)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'train'], loc='upper left')

    # Save locally
    directories.filecheck('./plots/A2/Early_Stopping/{}.png'.format(title1))
    directories.filecheck('./plots/A2/Early_Stopping/{}.png'.format(title2))
    plot_loss.savefig("./plots/A2/Early_Stopping/{}.png".format(title2), format='png')
    plot_acc.savefig("./plots/A2/Early_Stopping/{}.png".format(title1), format='png')

    # εκτύπωση αποτελεσμάτων
    print(f'Results sum (MSE) - Loss {loss_sum / 5} - Accuracy {acc_sum / 5}')
    f.close()
    sys.stdout = sys.__stdout__
    # καθαρισμός μνήμης
    print(f'Clearing session....')
    tensorflow.keras.backend.clear_session()
    plt.close(1)
    plt.close(2)
