import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import directories
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
import functions
import sys


print(f'Tensorflow version:{tensorflow.__version__}')
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

iterations = 9
gens = 1
# μέγεθος πληθυσμού
num_indiv = 19
# αριθμός γενεών
num_gen = 999
# πιθανότητα διασταύρωσης
crossrate = 0.6
# πιθανοτητα μετάλλαξης
mut = [0.00, 0.01, 0.10]
population = np.ones((num_indiv, 784))

for mutrate in mut:
    fname = "logs/PART_B/B2/results_{}_{}_{}.txt".format(num_indiv, crossrate, mutrate)
    directories.filecheck(fname)
    f = open(fname, 'w')
    sys.stdout = f
    for iter in range(iterations):
        big_avg = []
        best_results_gen = []
        # generate first population randomly
        population = np.random.randint(low=0, high=2, size=(num_indiv, 784))
        avg = []
        count = 0
        j = 0
        for gen in range(num_gen):
            print(f'Running generation number {gen} (iteration number {iter})')
            # test population fitness
            fit = functions.fitness(population, x_test, y_test)
            # fit contains the losses times the input amount for this generation's individuals
            fittest = np.min(fit)
            elit = population[np.where(fit == fittest)]
            best_results_gen.append(fittest)
            if gen > 0:
                if best_results_gen[j] == best_results_gen[j-1] or best_results_gen[j] < best_results_gen[j-1] - 0.01 * best_results_gen[j-1]:
                    count += 1
            avg.append(fittest)

            # select best individuals as parents
            parents = np.zeros(population.shape)
            for i in range(0, num_indiv):
                parents[i, :] = functions.select_parents(population, fit, 0.75)

            # crossover
            children = functions.mate(parents.astype(int), crossrate)
            # mutate
            mutated = functions.mutate(children, mutrate)

            population = mutated
            population[0, :] = elit[0, :]
            j += 1
            if count > 3:
                print(f'Μηδενική ή πολύ μικρή βελτίωση ατόμου, επόμενο τρέξιμο αλγορίθμου:')
                gens = gen
                break
            gens = gen

        # Μετά την ολοκλήρωση του for loop, θεωρητικά έχουμε το καλύτερο αποτέλεσμα πληθυσμού
        fit = functions.fitness(population, x_test, y_test)
        fittest = np.min(fit)
        avg.append(fittest)
        solution = population[np.where(fit == np.min(fit))]
        solution_idx = np.where(solution == 1)[0]
        average = 0
        for i in range(len(avg)):
            average += avg[i]
        average = average / gens
        print(f'{iter}Μέσος όρος απόδοσης για πληθυσμό μεγέθους {num_indiv}, πιθανότητα διασταύρωσης {crossrate} και πιθανότητα μετάλλαξης {mutrate} = {average}')
        big_avg.append(average)

        # plots
        plt_evolution = plt.figure(1)
        title = "population={}, c_rate={} και m_rate={}".format(num_indiv, crossrate, mutrate)
        plt.title(title, loc='center', pad=None)
        plt.plot(avg)
        plt.ylabel('Απόδοση')
        plt.xlabel('Γενεές')
        directories.filecheck('./plots/{}.png'.format(title))
        plt_evolution.savefig('./plots/{}.png'.format(title), format='png')
        plt.close(1)
        avg.clear()
        big_avg.clear()

    print(f'Clearing session....')
    # επιστροφή stdout στην κονσόλα
    f.close()
    sys.stdout = sys.__stdout__
    tensorflow.keras.backend.clear_session()
    plt.close(1)
