import single_layer, extra_layer, early_stopping
flag = True
print('Project Υπολογιστικής Νοημοσύνης - Μέρος Α')
while flag:
    rn = input('Επιλέξτε ερώτημα: \n2\n3\n4\n5\n')
    if rn == '2':
        rn_2 = input('Επιλέξτε υποερώτημα: \n6  (Single-layer MLP)\n7   (Two-Layer MLP)\n8  (Early-Stopping)\n')
        if rn_2 == '6':
            single_layer.single_layer()
        if rn_2 == '7':
            extra_layer.extra_layer()
    check = input('Συνέχεια με άλλο ερώτημα; (Y/N)')
    if check.casefold() == 'N':
        flag = False
