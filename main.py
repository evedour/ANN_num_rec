import single_layer
import extra_layer
import early_stopping
import A3
import A4

flag = True
print('Project Υπολογιστικής Νοημοσύνης - Μέρος Α')

while flag:
    rn = input('Επιλέξτε ερώτημα: \nA2\nA3\nA4\nA5\n\'all\''' to run all 7 files\n''')
    if rn == 'all':
        single_layer.single_layer()
        extra_layer.extra_layer()
        early_stopping.early_stopping()
        A3.a3()
        A4.a4()
    if rn == 'A2':
        rn_2 = input('Επιλέξτε υποερώτημα: \n1  (Single-layer MLP)\n2   (Two-Layer MLP)\n3  (Early-Stopping)\n')
        if rn_2 == '1':
            single_layer.single_layer()
        if rn_2 == '2':
            extra_layer.extra_layer()
        if rn_2 == '3':
            early_stopping.early_stopping()
    if rn == 'A3':
        A3.a4()
    if rn == 'A4':
        A4.A4()
    else:
        flag = False
    check = input('Συνέχεια με άλλο ερώτημα; (Y/N)')
    if check.casefold() == 'n':
        flag = False
