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

directories.B2()
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

# ορισμός των labels
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)


def myscaler(toscale):
    scaler = preprocessing.MinMaxScaler()
    for i in range(toscale.shape[0]):
        for j in range(toscale.shape[1]):
            toscale[i, j, :] = scaler.fit_transform(toscale[i, j, :])
    print(toscale)


def my_model(input_shape):
    # model
    # Δημιουργία μοντέλου με χρήση του keras API
    model = Sequential()
    # Πρώτο κρυφό επίπεδο
    model.add(Dense(384, input_shape=input_shape, activation='relu'))
    # Δεύτερο κρυφό επίπεδο
    model.add(Dense(50, activation='relu'))
    # Επίπεδο εξόδου
    model.add(Dense(10, activation='softmax'))
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


def score(bitstring):
    selected_features = []
    idx = 0
    for bit in range(bitstring):
        if bit == 1:
            selected_features.append(x_train[idx])
    idx += 1
    model = my_model((selected_features.shape, ))
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(lr=0.1, momentum=0.6,
                                                                                             decay=0.0, nesterov=False),
                  metrics=['accuracy'])
    fold = 1
    loss_sum = 0
    acc_sum = 0
    aval = []
    lval = []
    ltrain = []



def main():

    generation_size = 15
    population_size = 784
    # create population randomly
    population = [np.random.randint(0, 2, 9).tolist() for i in range(population_size)]

    for gen in range(generation_size):
        fname = './logs/B2/results_gen{}.txt'.format(gen)
        directories.filecheck(fname)
        f = open(fname, 'w')
        sys.stdout = f

        # fitness of population
        scores = [score(c) for c in population]

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
