import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import directories
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
import functions
import butler
import sys


print(f'Tensorflow version:{tensorflow.__version__}')
directories.B4()
num_indiv = 10
crossrate = 0.6
mutrate = 0.00

# GPU support
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print("CUDA - Αριθμός διαθέσιμων GPUs:", len(tensorflow.config.experimental.list_physical_devices('GPU')))
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# φόρτωση mnist από το keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# κάνουμε το mnist reshape
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
# x_train,test are now matrices of matrices

# MinMax scaling
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
# ορισμός των labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

user_in = input('Run B4-A? y/n ')
if user_in == 'y':
    fname = 'logs/B4/evaluation for best mlp.txt'
    directories.filecheck(fname)
    f = open(fname, 'w')
    sys.stdout = f
    butler.evaluate_best_mlp(num_indiv, crossrate, mutrate, x_test, y_test)
    # επιστροφή stdout στην κονσόλα
    f.close()
    sys.stdout = sys.__stdout__

user_in = input('Run B4-B? y/n ')
if user_in == 'y':
    x_train_selected, x_test_selected = butler.get_selected(x_train, x_test, num_indiv, crossrate, mutrate)
    # model
    # Δημιουργία μοντέλου με χρήση του keras API
    model = Sequential()
    # Πρώτο κρυφό επίπεδο
    model.add(Dense(794, input_shape=(784, ), activation='relu'))
    model.add(Masking(mask_value=0.0, input_shape=(x_train_selected.shape[0], )))
    # Δεύτερο κρυφό επίπεδο
    model.add(Dense(50, activation='relu'))
    # Επίπεδο εξόδου
    model.add(Dense(10, activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, decay=0.0, nesterov=False),
                  metrics=['accuracy'])

    fname = 'logs/B4/results after retraining mlp.txt'
    f = open(fname, 'w')
    sys.stdout = f

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
        history = model.fit(xi_train, yi_train, epochs=500, batch_size=200, verbose=1,
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
    title1 = 'Validation Accuracy'
    plt.title(title1, loc='center', pad=None)
    plt.plot(aval)
    plt.ylabel('acc')
    plt.xlabel('epoch')

    # loss
    plot_loss = plt.figure(2)
    title2 = 'Loss'
    plt.title(title2, loc='center', pad=None)
    # validation loss
    plt.plot(lval)
    # train loss
    plt.plot(ltrain)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'train'], loc='upper left')

    # Save locally
    directories.filecheck('./plots/{}.png'.format(title1))
    directories.filecheck('./plots/{}.png'.format(title2))
    plot_loss.savefig("./plots/{}.png".format(title2), format='png')
    plot_acc.savefig("./plots/{}.png".format(title1), format='png')

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
