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
gens = 1
# μέγεθος πληθυσμού
num_indiv = 20
# αριθμός γενεών
num_gen = 1000
# πιθανότητα διασταύρωσης
# cross = [0.9, 0.1]
crossrate = 0.6
# πιθανοτητα μετάλλαξης
mut = [0.00, 0.01, 0.10]
# mutrate = 0.01
population = np.ones((num_indiv, 784))

# μετρησεις:
# fit =  fitness scores για τη καθε γενια
# fittest = το καλύτερο σκορ στη γενια
# best results gen = λίστα των καλύτερων σκορ κάθε γενιάς για κάθε τρέξιμο του αλγορίθμου (χρησιμοποιείται στο να ελέγχεται το ποσοστό βελτίωσης σε κάθε τρέξιμο)
# big avg = λίστα όπου αποθηκεύονται οι μέσοι όροι αποδόσεων για κάθε τρέξιμο του αλγορίθμου

for mutrate in mut:
    # ανοιγμα αρχειου αποθηκευσης
    fname = "logs/PART_B/B2/results_{}_{}_{}.txt".format(num_indiv, crossrate, mutrate)
    directories.filecheck(fname)
    f = open(fname, 'w')
    sys.stdout = f
    # αρχικοποιησεις
    solution = np.empty((iterations, 784))
    solution_scores = np.empty((iterations, ))
    big_avg = []
    plt_evolution = plt.figure(1)
    for iter in range(iterations):
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
            fittest = np.max(fit)
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
        fittest = np.max(fit)
        avg.append(fittest)
        gens += 1

        # plot fitness evolution for best population for current iteration
        plt.plot(np.arange(len(fit)), fit, label=f'iteration {i}')
        plt.legend(f'iteration {iter}')
        average = 0
        for i in range(len(avg)):
            average += avg[i]
        average = average / gens
        print(f'Τρέξιμο νούμερο {iter} | Μέσος όρος απόδοσης για πληθυσμό μεγέθους {num_indiv}, πιθανότητα διασταύρωσης {crossrate} και πιθανότητα μετάλλαξης {mutrate} = {average}')
        big_avg.append(average)

    title1 = f"Fitness evolution, {num_indiv}_{crossrate}_{mutrate}"
    directories.filecheck('./plots/{}.png'.format(title1))
    plt_evolution.savefig('./plots/{}.png'.format(title1), format='png')
    plt.close(1)

    # plot averages for all iterations
    plt_avgs = plt.figure(2)
    title = "Μέσος όρος καλύτερων σκορ pop{} c_rate{} mut_rate{}".format(num_indiv, crossrate, mutrate)
    plt.title(title, loc='center', pad=None)
    plt.plot(big_avg)
    plt.ylabel('Score')
    plt.xlabel('Iteration')
    directories.filecheck('./plots/{}.png'.format(title))
    plt_avgs.savefig('./plots/{}.png'.format(title), format='png')
    plt.close(2)
    avg.clear()
    big_avg.clear()
    f_sol = "logs/PART_B/B2/solutions for {}_{}_{}.txt".format(num_indiv, crossrate, mutrate)
    f_fit = "logs/PART_B/B2/solution scores for {}_{}_{}.txt".format(num_indiv, crossrate, mutrate)
    directories.filecheck(f_sol)
    directories.filecheck(f_fit)

    # save solution for later use
    sol = population[np.where(fit == np.max(fit))]
    solution[iter, :] = sol[0, :].astype(int)
    solution_scores[iter] = np.amin(fit)
    np.save(f_sol, solution)
    np.save(f_fit, solution_scores)

    print(f'Clearing session....')
    # επιστροφή stdout στην κονσόλα
    f.close()
    sys.stdout = sys.__stdout__
    tensorflow.keras.backend.clear_session()
    plt.close(1)
