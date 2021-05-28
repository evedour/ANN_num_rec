import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
import directories
import sys


def myscaler(toscale):
    scaler = preprocessing.MinMaxScaler()
    for i in range(toscale.shape[0]):
        for j in range(toscale.shape[1]):
            toscale[i, j, :] = scaler.fit_transform(toscale[i, j, :])
    print(toscale)


def my_model(fl):
    print(f'Μοντέλο CNN με ένα επίπεδο 32 φίλτρων {5}x{5}, 2x2 MaxPooling, 20% dropout και MLP {fl}:10')
    # model
    model = Sequential()
    # first layer: Convolution 2D, 32 filters of size 5x5
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu', kernel_initializer='he_uniform'))
    # second layer: MaxPooling 2D, returns max value of image portion (2x2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # third layer: Flatten results of previous layers to feed into the MLP
    model.add(Flatten())
    # fourth and output layer: our standard MLP
    model.add(Dense(fl, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(10, kernel_initializer='he_uniform', activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def tourney(pop, scores, k=2):
    # select randomly from population and set as "best"
    selection = randint(len(pop))
    # iterate through population and find a better one
    for i in randint(0, len(pop), k - 1):
        if scores[i] < scores[selection]:
            selection = i
    return pop[selection]


def mate(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy
    if rand() < r_cross:
        # Σημείο τομής
        pt = randint(1, len(p1 - 2))
        c1 = p1[:pt] + p2[pt:]
        c2 = p1[pt:] + p2[:pt]
    return c1, c2


def mutate(bitstring, mutate_factor):
    for i in range(len(bitstring)):
        if rand() < mutate_factor:
            # flip the burger
            # Αυτό μπορεί και να γίνει με random assignment στα bits, απλά το flipping μου φαίνεται πιο "οργανωμένο"
            bitstring[i] = 1 - bitstring[i]


def main():
    # φόρτωση mnist από το keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # κάνουμε το mnist reshape
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # x_train,test are now matrices of matrices

    # MinMax scaling
    myscaler(x_test)
    myscaler(x_train)

    generation_size = 15
    population_size = 784

    population = [randint(0, 2, 9).tolist() for i in range(population_size)]
    # TODO: population size research
    for gen in range(generation_size):

        # fitness of population
        scores = [objective(c) for c in population]

        # select parents based on their fitness score
        selected = [tourney(pop, scores) for j in range(population_size)]

        # create next gen
        children = list()
        for i in range(0, population_size, 2):
            mom, dad = selected[i], selected[i + 1]
            for child in mate(mom, dad, 1.0):
                mutate(child, 0.1)
                children.append(child)

        population = children
