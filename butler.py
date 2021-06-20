import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import directories
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
import functions
import sys


def evaluate_best_mlp(num_indiv, crossrate, mutrate, x_test, y_test):
    model = functions.my_model((784,))
    solutions = np.load(f'logs/B2/numpy/solutions for {num_indiv}_{crossrate}_{mutrate}.npy')
    scores = np.load(f'logs/B2/numpy/solution scores for {num_indiv}_{crossrate}_{mutrate}.npy')
    best = solutions[np.where(scores == np.min(scores))]
    selected_test = functions.select_features(best, x_test)
    sze = np.where(best == 1)[0]
    penalty = sze.shape[0]
    print(f'Amount of selected features = {penalty}')
    results = model.evaluate(selected_test, y_test, verbose=1)
    print(f'Μετρικές για την καλύτερη επιλογή βάσει του GA: Accuracy = {results[1]} Loss = {results[0]}')


def get_selected(x_train, x_test, num_indiv, crossrate, mutrate):
    solutions = np.load(f'logs/B2/numpy/solutions for {num_indiv}_{crossrate}_{mutrate}.npy')
    scores = np.load(f'logs/B2/numpy/solution scores for {num_indiv}_{crossrate}_{mutrate}.npy')
    best = solutions[np.where(np.min(scores))]
    x_train = functions.select_features(best, x_train)
    x_test = functions.select_features(best, x_test)
    return x_train, x_test
