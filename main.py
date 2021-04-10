import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

features = 784
classes = 10

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
model.add(Dense(784, input_shape=input_shape, activation='relu'))
model.add(Dense(classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

fold = 1
kfold = KFold(5, shuffle=True, random_state=1)
for train, test in kfold.split(x_train):
    xi_train, xi_test = x_train[train], x_train[test]
    yi_train, yi_test = y_train[train], y_train[test]
    print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')
    fold = fold + 1
    model.fit(xi_train, yi_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2)
    #Test the model after training
    test_results = model.evaluate(xi_test, yi_test, verbose=1)
    print(f'Test results in fold # {fold} - Loss: {test_results[0]} - Accuracy: {test_results[1]}% - MSE {test_results[2]}')
