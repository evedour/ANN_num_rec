import directories
import tensorflow
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

def extra_layer():
    #αρχικοποίηση directories αποθήκευσης
    directories.extra_layer()

    #GPU support
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available", len(tensorflow.config.experimental.list_physical_devices('GPU')))
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    #αρχικοποίηση μεταβλητών
    features = 784
    classes = 10
    H1 = 794
    H2 = [10, 50, 100, 150, 200, 397]
    #κάνουμε το mnist reshape
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], features)
    x_test = x_test.reshape(x_test.shape[0], features)
    #κανονικοποίηση [0,1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    #ορισμός των labels
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)
    #ορισμός input shape για το μοντέλο MLP βάσει των χαρακτηριστικών
    input_shape = (features,)
    print(f'Feature shape: {input_shape}')

    #Έλεγχος για όλα τα H2
    for h_2 in H2:
        loss_sum = 0
        acc_sum = 0
        f_CE = "./logs/A2/Extra_Layer/results_CE_%s-%s.txt" % (H1, h_2)
        f_MSE ="./logs/A2/Extra_Layer/results_MSE_%s-%s.txt" % (H1, h_2)
        # Δημιουργία μοντέλων με χρήση του keras API
        model_CE = Sequential()
        model_MSE = Sequential()
        #πρώτο κρυφό επίπεδο
        model_CE.add(Dense(H1, input_shape=input_shape, activation='relu'))
        model_CE.add(Dense(h_2, activation='relu'))
        #δεύτερο κρυφό επίπεδο
        model_MSE.add(Dense(H1, input_shape=input_shape, activation='relu'))
        model_MSE.add(Dense(h_2, activation='relu'))
        #επίπεδο εξόδου
        model_CE.add(Dense(classes, activation='softmax'))
        model_MSE.add(Dense(classes, activation='softmax'))
        #compile
        #crossentropy
        model_CE.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        #mse
        model_MSE.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
        #αρχείο εξόδου
        f = open(f_CE, 'w')
        print('Starting CE for %s nodes in second layer' % h_2)
        sys.stdout = f
        #######################################################################################################################
        ###################################### CROSS ENTROPY 5-FOLD CV ########################################################
        fold = 1
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            #διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            #fit μοντέλου
            CE_history = model_CE.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1, validation_data=(xi_test, yi_test))

            #plots
            #accuracy
            plot_acc = plt.figure(1)
            title1 = 'Validation Accuracy Crossentropy Model {}-{}-10'.format(H1, h_2)
            plt.title(title1, loc='center', pad=None)
            plt.plot(CE_history.history['val_accuracy'])
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            #loss
            plot_loss = plt.figure(2)
            title2 ='Validation Loss Crossentropy Model {}-{}-10'.format(H1, h_2)
            plt.title(title2, loc='center', pad=None)
            plt.plot(CE_history.history['val_loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')


            #train loss
            plot_val = plt.figure(3)
            title3 = 'Training Loss Crossentropy Model {}-{}-10'.format(H1, h_2)
            plt.title(title3, loc='center', pad=None)
            plt.plot(CE_history.history['loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')


            #μετρήσεις μοντέλου
            CE_results = model_CE.evaluate(x_test, y_test, verbose=1)
            print(f'Test results in fold # {fold} - Loss: {CE_results[0]} - Accuracy: {CE_results[1]}')

            fold = fold + 1
            #αποθήκευση για προβολή των αποτελεσμάτων 5-fold CV
            loss_sum += CE_results[0]
            acc_sum += CE_results[1]
        # Save locally
        directories.filecheck('./plots/A2/Extra_Layer/{}.png'.format(title1))
        directories.filecheck('./plots/A2/Extra_Layer/{}.png'.format(title2))
        plot_loss.savefig("./plots/A2/Extra_Layer/{}.png".format(title2), format='png')
        plot_acc.savefig("./plots/A2/Extra_Layer/{}.png".format(title1), format='png')
        directories.filecheck('./plots/A2/Extra_Layer/{}.png'.format(title3))
        plot_val.savefig('./plots/A2/Extra_Layer/{}.png'.format(title3), format='png')
        #εκτυπωση αποτελεσμάτων
        print(f'Results sum (Crossentropy)- Loss {loss_sum/5} - Accuracy {acc_sum/5}')
        #αναμονή input για την αποθήκευση των μετρήσεων
        f.close()
        sys.stdout = sys.__stdout__
        #απελευθερωση μνημης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)
        plt.close(3)
        #αρχικοποίηση καινούριων μεταβλητων
        loss_sum = 0
        acc_sum = 0
        fold = 1
        #νεο αρχείο εξόδου
        f = open(f_MSE, 'w')
        print('Starting MSE for %s nodes' % h_2)
        sys.stdout = f
        #######################################################################################################################
        #################################### MEAN SQUARED ERROR 5-FOLD CV #####################################################
        kfold = KFold(5, shuffle=True, random_state=1)
        for train, test in kfold.split(x_train):
            # διαχωρισμός train-test indexes
            xi_train, xi_test = x_train[train], x_train[test]
            yi_train, yi_test = y_train[train], y_train[test]
            print(f' fold # {fold}, TRAIN: {train}, TEST: {test}')

            # fit μοντέλου
            MSE_history = model_MSE.fit(xi_train, yi_train, epochs=10, batch_size=200, verbose=1, validation_data=(xi_test, yi_test))

            # plots
            # accuracy
            plot_acc = plt.figure(1)
            title1 = 'Validation Accuracy MSE Model {}-{}-10'.format(H1, h_2)
            plt.title(title1, loc='center', pad=None)
            plt.plot(MSE_history.history['val_accuracy'])
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # loss
            plot_loss = plt.figure(2)
            title2 = 'Validation Loss MSE Model {}-{}-10'.format(H1, h_2)
            plt.title(title2, loc='center', pad=None)
            plt.plot(MSE_history.history['val_loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # train loss
            plot_val = plt.figure(3)
            title3 = 'Training Loss MSE Model {}-{}-10'.format(H1, h_2)
            plt.title(title3, loc='center', pad=None)
            plt.plot(MSE_history.history['loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'], loc='upper left')

            # μετρήσεις μοντέλου
            MSE_results = model_MSE.evaluate(x_test, y_test, verbose=1)
            print(f'Test results in fold # {fold} - Loss: {MSE_results[0]} - Accuracy: {MSE_results[1]}')

            fold = fold + 1
            # αποθήκευση για προβολή των αποτελεσμάτων 5-fold CV
            loss_sum += MSE_results[0]
            acc_sum += MSE_results[1]
        #Sace locally
        directories.filecheck('./plots/A2/Extra_Layer/{}.png'.format(title1))
        directories.filecheck('./plots/A2/Extra_Layer/{}.png'.format(title2))
        plot_loss.savefig("./plots/A2/Extra_Layer/{}.png".format(title2), format='png')
        plot_acc.savefig("./plots/A2/Extra_Layer/{}.png".format(title1), format='png')
        directories.filecheck('./plots/A2/Extra_Layer/{}.png'.format(title3))
        plot_val.savefig('./plots/A2/Extra_Layer/{}.png'.format(title3), format='png')
        #εκτύπωση αποτελεσμάτων
        print(f'Results sum (MSE) - Loss {loss_sum/5} - Accuracy {acc_sum/5}')
        f.close()
        sys.stdout = sys.__stdout__
        #καθαρισμός μνήμης
        print(f'Clearing session....')
        tensorflow.keras.backend.clear_session()
        plt.close(1)
        plt.close(2)
        plt.close(3)