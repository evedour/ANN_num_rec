import tensorflow
import numpy as np
import math
import directories
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing


def my_model(input_shape):
    # model
    # Δημιουργία μοντέλου με χρήση του keras API
    model = Sequential()
    # Πρώτο κρυφό επίπεδο
    model.add(Dense(794, input_shape=input_shape, activation='relu'))
    # Δεύτερο κρυφό επίπεδο
    model.add(Dense(50, activation='relu'))
    # Επίπεδο εξόδου
    model.add(Dense(10, activation='softmax'))
    # compile the model
    model.load_weights('my_model_weights.h5')
    # model = tensorflow.keras.models.load_model('my_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(lr=0.1, momentum=0.6, decay=0.0, nesterov=False),
                  metrics=['accuracy'])

    return model


def select_features(chromosome, features):
    # Find the indexes where we keep the features and create a new feature vector
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
        print(f'Penalty={penalty}')
        model = my_model((784, ))
        # model.fit(selected_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2, callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])
        results = model.evaluate(selected_test, y_test, verbose=1)
        # fitness func
        # scores[i] = results[0] + math.exp(penalty)
        scores[i] = results[0] + ( -1 * ((1/(math.sqrt(2*3.14*1.0))) * (-pow((penalty-392), 2)/(2*1.0))))
        print(f'Αποτελέσματα στο άτομο {i}: loss = {results[0]}, score = {scores[i]}')
        i = i + 1
        results.clear()

    return scores


def select_parents(population, fit, amount):
    fittest = np.empty((amount, population.shape[1]))
    for parent in range(amount):
        parent_idx = np.where(fit == np.min(fit))
        parent_idx = parent_idx[0][0]
        fittest[parent, :] = population[parent_idx, :]
        fit[parent_idx] = 99999999999

    return fittest


def mate(par, crossrate, amount):
    child = np.empty(amount) # 15 x 784
    for i in range(par.shape[0]):
        # crossover point
        crosspoint = np.random.randint(1, len(par[i])-2)
        if np.random.rand() < crossrate:
            # crossover is performed
            child[i, 0:crosspoint] = par[i % par.shape[0], 0:crosspoint]
            child[i, crosspoint:] = par[(i+1) % par.shape[0], crosspoint:]

    return child


def mutate(individuals, mutrate):
    mutation_idx = np.random.randint(low=0, high=individuals.shape[1], size=2)
    for i in range(individuals.shape[0]):
        if np.random.rand() < mutrate:
            individuals[i, mutation_idx] = 1 - individuals[i, mutation_idx]

    return individuals
