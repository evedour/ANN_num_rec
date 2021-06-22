import tensorflow
import numpy as np
import math
import directories
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Masking
from sklearn import preprocessing
from math import exp


def my_model(input_shape):
    # model
    # Δημιουργία μοντέλου με χρήση του keras API
    model = Sequential()
    # Πρώτο κρυφό επίπεδο
    model.add(Dense(794, input_shape=(784, ), activation='relu'))
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    # Δεύτερο κρυφό επίπεδο
    model.add(Dense(50, activation='relu'))
    # Επίπεδο εξόδου
    model.add(Dense(10, activation='softmax'))
    # compile the model
    model.load_weights('my_model_weights.h5')
    # model = tensorflow.keras.models.load_model('my_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, decay=0.0, nesterov=False),
                  metrics=['accuracy'])

    return model


def select_features(chromosome, features):
    # Εύρεση δεικτών προς τα χαρακτηριστικά που κρατάμε
    idx = np.where(chromosome == 1)[0]
    selected_feats = np.zeros(features.shape)
    selected_feats[:, idx] = selected_feats[:, idx] + features[:, idx]
    return selected_feats


def fitness(population, x_test,  y_test):
    i = 0
    scores = np.zeros(population.shape[0])
    for individual in population:
        selected_test = select_features(individual, x_test)
        sze = np.where(individual == 1)[0]
        penalty = sze.shape[0]
        print(f'Amount of nodes used = {penalty}')
        model = my_model((penalty, ))
        results = model.evaluate(selected_test, y_test, verbose=1)
        # fitness func
        x = penalty/784
        scores[i] = results[0] + 1/(1 + exp(-(x - 0.5)*10))
        print(f'Αποτελέσματα στο άτομο {i}: loss = {results[0]}, score = {scores[i]}')
        i = i + 1
        results.clear()

    return scores


def select_parents(population, fit, k):
    selection_i = np.random.randint(population.shape[0])
    # επιλογή 2 τυχαίων γονέων
    for i in np.random.randint(0, population.shape[0], 2):
        # τουρνουά ( ο πιο fit επιλέγεται )
        if np.random.random() < k:
            if fit[i] < fit[selection_i]:
                selection_i = i
        else:
            if fit[i] > fit[selection_i]:
                selection_i = i

    return population[selection_i]


def mate(par, crossrate):
    child = np.zeros(par.shape).astype(int)
    for i in range(par.shape[0]):
        # crossover point
        crosspoint = np.random.randint(1, len(par[i]) - 2)
        if np.random.rand() < crossrate:
            # crossover is performed
            child[i, 0:crosspoint] = par[i % par.shape[0], 0:crosspoint]
            child[i, crosspoint:] = par[(i + 1) % par.shape[0], crosspoint:]

    return child


def mutate(individuals, mutrate):
    mutation_idx = np.random.randint(low=0, high=individuals.shape[1], size=2)
    for i in range(individuals.shape[0]):
        if np.random.rand() < mutrate:
            individuals[i, mutation_idx] = 1 - individuals[i, mutation_idx]

    return individuals
