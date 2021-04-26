import A5
import A5_dropout
import single_layer
import extra_layer
import early_stopping
import A3
import A4

flag = True
epochs = 250
print('Project Υπολογιστικής Νοημοσύνης - Μέρος Α')

while flag:
    rn = input('Επιλέξτε ερώτημα: \nA2\nA3\nA4\nA5\n\'all\''' to run all 7 files\n''')
    if rn == 'all':
        single_layer.single_layer(epochs)
        extra_layer.extra_layer(epochs)
        early_stopping.early_stopping(epochs)
        A3.a3(epochs)
        A4.a4(epochs)
    if rn == 'A2':
        rn_2 = input('Επιλέξτε υποερώτημα: \n1  (Single-layer MLP)\n2   (Two-Layer MLP)\n3  (Early-Stopping)\n')
        if rn_2 == '1':
            single_layer.single_layer(epochs)
            flag = False
        if rn_2 == '2':
            extra_layer.extra_layer(epochs)
            flag = False
        if rn_2 == '3':
            early_stopping.early_stopping(epochs)
            flag = False
    if rn == 'A3':
        A3.a3(epochs)
        flag = False
    if rn == 'A4':
        A4.a4(epochs)
        flag = False
    if rn == 'A5':
        rn_3 = input('Με ή χωρίς επίπεδο dropout; ')
        if rn_3 == 'με':
            A5_dropout.a5()
            flag = False
        else:
            A5.a5()
            flag = False
    else:
        flag = False
    check = input('Συνέχεια με άλλο ερώτημα; (Y/N)')
    if check == 'Y':
        flag = True
