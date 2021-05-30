import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import functions
import directories


directories.B2()
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

# number of individuals in a population
num_indiv = 10
# number of generations
num_gen = 5

# TODO sparse matrix
population = np.random.randint(low=0, high=2, size=(num_indiv, 784))
scores = []
x = []

fit = functions.fitness(population, x_train, x_test, y_train, y_test)
