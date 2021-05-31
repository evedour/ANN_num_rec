import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import directories
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import functions

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
# generate first population randomly
population = np.random.randint(low=0, high=2, size=(num_indiv, 784))

for gen in range(num_gen):
    # test population fitness
    fit = functions.fitness(population, x_train, x_test, y_train, y_test)
    # fit contains the losses for this generation's individuals

    # select best individuals as parents
    parents = functions.select_parents(population, fit, 5)

    # crossover
    children = []
    for i in len(parents):
        if i < len(parents):
            children.append(functions.mate(parent[i], parent[i+1], 1.0))
        else:
            break;

    mutated = functions.mutate(children, 0.1)

    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = mutated
