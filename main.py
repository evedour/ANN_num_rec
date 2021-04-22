import single_layer, extra_layer, early_stopping, A3, A4, A5, A5_dropout

flag = True
print('Project Υπολογιστικής Νοημοσύνης - Μέρος Α')
while(flag):
    rn = input('Επιλέξτε ερώτημα: 2, 3, 4, 5 ')
    if(rn == '2'):
        rn_2 = input('Επιλέξτε υποερώτημα: 6(Single-layer MLP),7(Two-Layer MLP),8(Early-Stopping)')
        if(rn_2 == '6'):
            single_layer.single_layer()
        if(rn_2 == '7'):
            extra_layer.extra_layer()
    check = input('Συνέχεια με άλλο ερώτημα; (Y/N)')
    if(check == 'N'):
        flag = False
