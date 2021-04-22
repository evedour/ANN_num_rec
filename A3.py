import tensorflow
import directories
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

def A3():
    # αρχικοποίηση directories αποθήκευσης
    directories.A3()

    # GPU support
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    # αρχικοποίηση μεταβλητών
    features = 784
    classes = 10
    H1 = 794
    H2 = 10
    loss_fun = 'categorical_crossentropy'
    learning_rates = [0.0001, 0.0001, 0.005, 0.1]
    momentums = [0.2, 0.6]
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

    # Δημιουργία μοντέλου με χρήση του keras API
    model = Sequential()
    # πρώτο κρυφό επίπεδο
    model.add(Dense(H1, input_shape=input_shape, activation='relu'))
    model.add(Dense(H2, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    for i in range(learning_rates):
        if i == 0:
            m=0.2
            opt = tensorflow.keras.optimizers.SGD(lr=learning_rates[i], momentum = 0.2, decay=0.0, nesterov=False)
            fname = './logs/A3/results_{}_{}.txt'.format(learning_rates[i], 0.2)
        else:
            m=0.6
            opt = tensorflow.keras.optimizers.SGD(lr=learning_rates[i], momentum = 0.6, decay=0.0, nesterov=False)
            fname = './logs/A3/results_{}_{}.txt'.format(learning_rates[i], 0.6)

        model.compile(loss=loss_fun, optimizer=opt, metrics=['accuracy'],)
        fold = 1
        loss_sum = 0
        acc_sum = 0
        f = open(fname, 'w')
        sys.stdout = f
        #5-fold CV
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            # διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')
            # fit μοντέλου
            history = model.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1,
                                validation_data=(xi_test, yi_test),
                                callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])
            # plots
            # accuracy
            plt_acc = plt.figure(1)
            title1 = 'Validation Accuracy, learning rate = {}, momentum = {}'.format(learning_rates[i], m)
            plt.title(title1, loc='center', pad=None)
            plt.plot(history.history['val_accuracy'])
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # loss
            plt_loss = plt.figure(2)
            plt.plot(history.history['val_loss'])
            title2 = 'Validation Loss, learning rate = {}, momentum = {}'.format(learning_rates[i], m)
            plt.title(title2, loc='center', pad=None)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # train loss
            plt_val = plt.figure(3)
            title3 = 'Train Loss, learning rate = {}, momentum = {}'.format(learning_rates[i], m)
            plt.title(title3, loc='center', pad=None)
            plt.plot(history.history['loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # Test the model after training
            test_results = model.evaluate(x_test, y_test, verbose=1)
            print(f'Test results in fold # {fold} - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
            fold = fold + 1
            # save 5-fold cv results
            loss_sum += test_results[0]
            acc_sum += test_results[1]
        # Save locally
        directories.filecheck('./plots/A3/{}.png'.format(title1))
        plt_acc.savefig('./plots/A3/{}.png'.format(title1), format='png')
        directories.filecheck('./plots/A3/{}.png'.format(title2))
        plt_loss.savefig('./plots/A3/{}.png'.format(title2), format='png')
        directories.filecheck('./plots/A3/{}.png'.format(title3))
        plt_loss.savefig('./plots/A3/{}.png'.format(title3), format='png')
        print(f'Results sum - Loss {loss_sum / 5} - Accuracy {acc_sum / 5}')
        f.close()
        sys.stdout = sys.__stdout__
        # απελευθερωση μνημης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)
        plt.close(3)