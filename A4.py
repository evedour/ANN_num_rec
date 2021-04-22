import sys

import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
import directories


def A4():
    # αρχικοποίηση directories αποθήκευσης
    directories.A4()

    # check tensorflow's GPU support
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    features = 784
    classes = 10
    H1 = 794
    H2 = 10
    loss_fun = 'categorical_crossentropy'
    learning_rate = 0.1
    m = 0.6
    L2 = [0.1, 0.5, 0.9]
    # κάνουμε το mnist reshape
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], features)
    x_test = x_test.reshape(x_test.shape[0], features)

    # κανονικοποίηση [0,1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # ορισμός των labels
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # ορισμός input shape για το μοντέλο MLP βάσει των χαρακτηριστικών
    input_shape = (features,)
    print(f'Feature shape: {input_shape}')
    opt = tensorflow.keras.optimizers.SGD(lr = learning_rate, momentum = m, decay=0.0, nesterov=False)
    for l2_factor in L2:
        # Create the model
        fname = './logs/A4/results_{}'.format(l2_factor)
        model = Sequential()
        model.add(Dense(H1, kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor), input_shape=input_shape, activation='relu'))
        model.add(Dense(H2, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss=loss_fun, optimizer=opt, metrics='accuracy')

        fold = 1
        loss_sum = 0
        acc_sum = 0
        f = open(fname, 'w')
        sys.stdout = f
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            history = model.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1,
                                validation_data=(xi_test, yi_test),
                                callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])
            # plots
            # accuracy
            plt_acc = plt.figure(1)
            title1 = 'Validation Accuracy, L2 factor = {}'.format(l2_factor)
            plt.title(title1)
            plt.plot(history.history['val_accuracy'])
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # loss
            plt_loss = plt.figure(2)
            title2 = 'Validation Loss, L2 factor = {}'.format(l2_factor)
            plt.title(title2)
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # train loss
            plot_val = plt.figure(3)
            title3 = 'Validation Accuracy, L2 factor = {}'.format(l2_factor)
            plt.title(title3)
            plt.title('Training Loss', loc='center', pad=None)
            plt.plot(history.history['loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # Test the model after training
            test_results = model.evaluate(xi_test, yi_test, verbose=1)
            print(f'Test results in fold # {fold} - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
            fold = fold + 1
            # save 5-fold cv results
            loss_sum += test_results[0]
            acc_sum += test_results[1]
        #Save locally
        directories.filecheck('./plots/A4/{}.png'.format(title1))
        plt_acc.savefig('./plots/A4/{}.png'.format(title1), format='png')
        directories.filecheck('./plots/A4/{}.png'.format(title2))
        plt_loss.savefig('./plots/A4/{}.png'.format(title2), format='png')
        directories.filecheck('./plots/A4/{}.png'.format(title3))
        plt_loss.savefig('./plots/A4/{}.png'.format(title3), format='png')
        print(f'Results sum - Loss {loss_sum/ 5} - Accuracy {acc_sum / 5}')
        f.close()
        sys.stdout = sys.__stdout__
        # απελευθερωση μνημης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)
        plt.close(3)