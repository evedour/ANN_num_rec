import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import directories
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing
import functions
import butler
import sys
import time
import os

print(f'Tensorflow version:{tensorflow.__version__}')
directories.B2()
start_time = time.time()
np.set_printoptions(threshold=np.inf)
# GPU support
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print("CUDA - Αριθμός διαθέσιμων GPUs:", len(tensorflow.config.experimental.list_physical_devices('GPU')))
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

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

iterations = 10
gens = 0
# μέγεθος πληθυσμού
no_of_individuals = [20, 200]
# αριθμός γενεών
num_gen = 1000

tweaks = [(0.6, 0.00), (0.6, 0.01), (0.6, 0.10), (0.9, 0.01), (0.1, 0.01)]

# μετρήσεις:
# fit =  fitness scores για τη κάθε γενιά
# fittest = το καλύτερο σκορ στη γενιά
# best results gen = λίστα των καλύτερων σκορ κάθε γενιάς για κάθε τρέξιμο του αλγορίθμου (χρησιμοποιείται στο να ελέγχεται το ποσοστό βελτίωσης σε κάθε τρέξιμο)
# fitness_history = κάθε γραμμή πίνακα αποθηκεύει ανα γενιά την καλύτερη απόδοση του αλγορίθμου στο αντίστοιχο τρέξιμο

for num_indiv in no_of_individuals:

    population = np.ones((num_indiv, 784))

    for crossrate, mutrate in tweaks:

        # άνοιγμα αρχείου αποθήκευσης
        fname = "logs/B2/results_{}_{}_{}.txt".format(num_indiv, crossrate, mutrate)
        directories.filecheck(fname)
        f = open(fname, 'w')
        sys.stdout = f

        # αρχικοποιήσεις
        solution = np.empty((iterations, 784))
        solution_scores = np.empty((iterations, ))
        fitness_history = np.zeros((10, num_gen))
        gens_needed = []
        average = 0

        # GA
        for iter in range(iterations):
            best_results_gen = []

            # ο αρχικός πληθυσμός παράγεται τυχαία
            population = np.random.randint(low=0, high=2, size=(num_indiv, 784))
            count = 0
            j = 0

            for gen in range(num_gen-1):
                print(f'Running generation number {gen} (iteration number {iter})')
                # population fitness
                fit = functions.fitness(population, x_test, y_test)
                fittest = np.min(fit)
                elit = population[np.where(fit == fittest)]
                fitness_history[iter, gen] = fittest
                best_results_gen.append(fittest)
                if gen > 0:
                    if best_results_gen[j] == best_results_gen[j-1] or best_results_gen[j] < best_results_gen[j-1] - 0.01 * best_results_gen[j-1]:
                        count += 1

                # επιλογή γονέων
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
                if count > 5:
                    print(f'Μηδενική ή πολύ μικρή βελτίωση ατόμου, επόμενο τρέξιμο αλγορίθμου:')
                    gens = gen
                    break
                gens = gen

            # Μετά την ολοκλήρωση του for loop, θεωρητικά έχουμε το καλύτερο αποτέλεσμα πληθυσμού
            fit = functions.fitness(population, x_test, y_test)
            fittest = np.min(fit)
            fitness_history[iter, gen+1] = fittest
            gens += 1
            gens_needed.append(gens)
            average += fittest

        print(f'Αποτελέσματα για πληθυσμό {num_indiv} ατόμων, με crossover rate = {crossrate} και mutation rate = {mutrate}: Μέσος όρος απόδοσης βέλτιστου ανα τρέξιμο= {average/10}')
        print(f'Κατά μέσο όρο χρειάστηκαν {sum(gens_needed) / 10} γενιές')
        print(f'This whole thing took me {(time.time() - start_time)} seconds or {(time.time() - start_time)/60} minutes')

        evolution = []
        for i in range(max(gens_needed)):
            evolution.append(np.mean(fitness_history[:, i], axis=0))

        # επιστροφή stdout στην κονσόλα
        f.close()
        sys.stdout = sys.__stdout__

        # PLOTS
        plt_evolution = plt.figure(1)
        title = f"Εξέλιξη πληθυσμού {num_indiv} ατόμων για c_rate = {crossrate} και m_rate = {mutrate}"
        plt.title(title, loc='center', pad=None)
        plt.plot(evolution)
        plt.ylabel('Απόδοση')
        plt.xlabel('Γενιά')
        directories.filecheck('./plots/{}.png'.format(title))
        plt_evolution.savefig('./plots/{}.png'.format(title), format='png')

        # SOLUTIONS
        f_sol = "logs/B2/numpy/solutions for {}_{}_{}".format(num_indiv, crossrate, mutrate)
        f_fit = "logs/B2/numpy/solution scores for {}_{}_{}".format(num_indiv, crossrate, mutrate)
        directories.filecheck(f_sol)
        directories.filecheck(f_fit)
        # save solution for later use
        sol = population[np.where(fit == np.min(fit))[0], :]
        solution = sol.astype(int)
        solution_scores = np.amin(fit)
        np.save(f_sol, solution)
        np.save(f_fit, solution_scores)

        print(f'Clearing session....')

        tensorflow.keras.backend.clear_session()
        plt.close(1)


os.system("shutdown /s /t 60")
