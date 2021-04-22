flag = True
print('Project Υπολογιστικής Νοημοσύνης - Μέρος Α')
while(flag):
    rn = input('Επιλέξτε ερώτημα: 2, 3, 4, 5 ')
    if(rn == '2'):
        rn_2 = input('Επιλέξτε υποερώτημα: 6(Single-layer MLP),7(Two-Layer MLP),8(Early-Stopping)')
        if(rn_2 == '6'):
            import single_layer
            single_layer.single_layer()
        if(rn_2 == '7'):
            import extra_layer
            extra_layer.extra_layer()
    check = input('Συνέχεια με άλλο ερώτημα; (Y/N)')
    if(check == 'N'):
        flag = False
