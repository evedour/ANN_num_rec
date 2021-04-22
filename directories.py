import os.path

parent = os.path.dirname(os.path.abspath(__file__))
#A2
def single_layer():
    #logs
    if not(os.path.isfile('%s/logs' % parent)):
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isfile('%s/logs/A2' %parent)):
        folder_name = 'A2'
        folder = os.path.join('%s/logs' % parent, folder_name)
        os.mkdir(folder)
    folder_name = 'Single_Layer'
    folder = os.path.join('%s/logs/A2' % parent, folder_name)
    os.mkdir(folder)
    #plots
    if not(os.path.isfile('%s/plots' % parent)):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isfile('%s/plots/A2' %parent)):
        folder_name = 'A2'
        folder = os.path.join('%s/plots' % parent, folder_name)
        os.mkdir(folder)
    folder_name = 'Single_Layer'
    folder = os.path.join('%s/plots/A2' % parent, folder_name)
    os.mkdir(folder)
#######################################################################################################################