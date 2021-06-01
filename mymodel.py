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


# αρχικοποίηση μεταβλητών
features = 784
classes = 10
loss_f = ['categorical_crossentropy']
h1 = 794
h2 = 50
m = 0.6
learning_rates = [0.1]

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
for loss_fun in loss_f:
    # Δημιουργία μοντέλου με χρήση του keras API
    model = Sequential()
    # Πρώτο κρυφό επίπεδο
    model.add(Dense(h1, input_shape=input_shape, activation='relu'))
    # Δεύτερο κρυφό επίπεδο
    model.add(Dense(h2, activation='relu'))
    # Επίπεδο εξόδου
    model.add(Dense(classes, activation='softmax'))
    i = 0
    for lrate in learning_rates:
        print(f'Set SGD optimizer to learning rate={lrate} and momentum={m}')
        opt = tensorflow.keras.optimizers.SGD(lr=lrate, momentum=m, decay=0.0, nesterov=False)
        # compile
        model.compile(loss=loss_fun, optimizer=opt, metrics=['accuracy'])

        fold = 1
        loss_sum = 0
        acc_sum = 0
        aval = []
        lval = []
        ltrain = []

        # 5-fold CV
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            # διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            # fit μοντέλου
            history = model.fit(xi_train, yi_train, epochs=100, batch_size=200, verbose=1,
                                validation_data=(xi_test, yi_test),
                                callbacks=[
                                    tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])

            # στατιστικά
            aval.append(np.mean(history.history['val_accuracy']))
            lval.append(np.mean(history.history['val_loss']))
            ltrain.append(np.mean(history.history['loss']))

            # μετρησεις μοντέλου
            results = model.evaluate(x_test, y_test, verbose=1)
            print(f'Αποτελέσματα στο fold # {fold} - Loss: {results[0]} - Accuracy: {results[1]}')

            fold += 1
            # αποθήκευση για προβολή των αποτελεσμάτων 5-fold CV
            loss_sum += results[0]
            acc_sum += results[1]

        # εκτύπωση αποτελεσμάτων
        print(f'Συνολικά Αποτελέσματα - Loss {loss_sum / 5} - Accuracy {acc_sum / 5}')
        model.save_weights('./my_model_weights.h5')
        tensorflow.keras.models.save_model(model, "my_model.h5", save_format="h5")
        # απελευθερωση μνημης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()



