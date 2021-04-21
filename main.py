import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model

features = 784
classes = 10
entropy_sum = 0
acc_sum = 0
mse_sum = 0
#reshape and prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], features)
x_test = x_test.reshape(x_test.shape[0], features)

#normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#labels
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

input_shape = (features,)
print(f'Feature shape: {input_shape}')
# Create the model
model = Sequential()
#add the hidden layer
model.add(Dense(794, input_shape=input_shape, activation='relu'))
#add output layer
model.add(Dense(classes, activation='softmax'))
#set the metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

fold = 1
kfold = KFold(5, shuffle=True, random_state=1)
for train, test in kfold.split(x_train):
    xi_train, xi_test = x_train[train], x_train[test]
    yi_train, yi_test = y_train[train], y_train[test]
    print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

    history = model.fit(xi_train, yi_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2)
    #plots
    #accuracy
    plot_acc = plt.figure(1)
    plt.title('Validation Accuracy', loc='center', pad=None)
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

    #loss
    plot_loss = plt.figure(2)
    plt.title('Validation Loss', loc='center', pad=None)
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

    #train loss
    plot_val = plt.figure(3)
    plt.title('Training Loss', loc='center', pad=None)
    plt.plot(history.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

    #Test the model after training
    test_results = model.evaluate(xi_test, yi_test, verbose=1)
    plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

    print(f'Test results in fold # {fold} - Loss: {test_results[0]} - Accuracy: {test_results[1]}% - MSE {test_results[2]}')
    fold = fold + 1
    #save 5-fold cv results
    entropy_sum += test_results[0]
    acc_sum += test_results[1]
    mse_sum += test_results[2]

plt.show()
print(f'Results sum - Loss {entropy_sum/5} - Accuracy {acc_sum/5}%- MSE {mse_sum/5}')
