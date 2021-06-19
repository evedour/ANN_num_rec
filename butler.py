import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import directories
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
import functions
import sys


def evaluate_best_mlp(num_indiv, crossrate, mutrate, x_test, y_test):
    model = functions.my_model((784,))
    solutions = np.load(f'solutions for {num_indiv}_{crossrate}_{mutrate}.txt.npy')
    scores = np.load(f'solution scores for {num_indiv}_{crossrate}_{mutrate}.txt.npy')
    best = solutions[np.where(np.max(scores))]
    selected_test = functions.select_features(best, x_test)
    results = model.evaluate(selected_test, y_test, verbose=1)
    print(f'Μετρικές για την καλύτερη επιλογή βάσει του GA: Accuracy = {results[1]} Loss = {results[0]}')


def get_selected(x_train, x_test):
    solutions = np.load(f'solutions for {num_indiv}_{crossrate}_{mutrate}.txt.npy')
    scores = np.load(f'solution scores for {num_indiv}_{crossrate}_{mutrate}.txt.npy')
    best = solutions[np.where(np.max(scores))]
    x_train = functions.select_features(best, x_train)
    x_test = functions.select_features(best, x_test)
    return x_train, x_test
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
                history = model.fit(xi_train, yi_train, epochs=ep, batch_size=200, verbose=1,
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
            title1 = 'Validation Accuracy, {} model r={}'.format(loss_fun, factor)
            plt.title(title1, loc='center', pad=None)
            plt.plot(aval)
            plt.ylabel('acc')
            plt.xlabel('epoch')

            # loss
            plot_loss = plt.figure(2)
            title2 = 'Loss, {} model r={}'.format(loss_fun, factor)
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