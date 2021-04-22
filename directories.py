import os
import fnmatch

parent = os.path.dirname(os.path.abspath(__file__))

def filecheck(filename):
    for root, dirs, files in os.walk(parent):
        for dr in dirs:
            os.path.join(root, dr)
        for name in files:
            if fnmatch.fnmatch(name, filename):
                os.remove(os.path.join(root, filename))

#A2
def single_layer():
    #logs
    if not(os.path.isdir('{}/logs'.format(parent))):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/logs/A2'.format(parent))):
        folder_name = 'A2'
        folder = os.path.join('{}/logs'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/logs/A2/Single_Layer'.format(parent))):
        folder_name = 'Single_Layer'
        folder = os.path.join('{}/logs/A2'.format(parent), folder_name)
        os.mkdir(folder)
    #plots
    if not(os.path.isdir('{}/plots'.format(parent))):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/plots/A2'.format(parent))):
        print('creating plot logs...')
        folder_name = 'A2'
        folder = os.path.join('{}/plots'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/plots/A2/Single_Layer'.format(parent))):
        folder_name = 'Single_Layer'
        folder = os.path.join('{}/plots/A2'.format(parent), folder_name)
        os.mkdir(folder)
#######################################################################################################################
def extra_layer():
    #logs
    if not(os.path.isdir('{}/logs'.format(parent))):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/logs/A2'.format(parent))):
        folder_name = 'A2'
        folder = os.path.join('{}/logs'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/logs/A2/Extra_Layer'.format(parent))):
        folder_name = 'Extra_Layer'
        folder = os.path.join('{}/logs/A2'.format(parent), folder_name)
        os.mkdir(folder)
    #plots
    if not(os.path.isdir('{}/plots'.format(parent))):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/plots/A2'.format(parent))):
        print('creating plot logs...')
        folder_name = 'A2'
        folder = os.path.join('{}/plots'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/plots/A2/Extra_Layer'.format(parent))):
        folder_name = 'Extra_Layer'
        folder = os.path.join('{}/plots/A2'.format(parent), folder_name)
        os.mkdir(folder)
########################################################################################################################
def extra_layer():
    #logs
    if not(os.path.isdir('{}/logs'.format(parent))):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/logs/A2'.format(parent))):
        folder_name = 'A2'
        folder = os.path.join('{}/logs'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/logs/A2/Early_Stopping'.format(parent))):
        folder_name = 'Early_Stopping'
        folder = os.path.join('{}/logs/A2'.format(parent), folder_name)
        os.mkdir(folder)
    #plots
    if not(os.path.isdir('{}/plots'.format(parent))):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/plots/A2'.format(parent))):
        print('creating plot logs...')
        folder_name = 'A2'
        folder = os.path.join('{}/plots'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/plots/A2/Early_Stopping'.format(parent))):
        folder_name = 'Early_Stopping'
        folder = os.path.join('{}/plots/A2' .format(parent), folder_name)
        os.mkdir(folder)
########################################################################################################################
########################################################################################################################
#A3
def A3():
    # logs
    if not (os.path.isdir('{}/logs'.format(parent))):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/logs/A3'.format(parent))):
        folder_name = 'A3'
        folder = os.path.join('{}/logs'.format(parent), folder_name)
        os.mkdir(folder)
    # plots
    if not (os.path.isdir('{}/plots'.format(parent))):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/plots/A3'.format(parent))):
        print('creating plot logs...')
        folder_name = 'A3'
        folder = os.path.join('{}/plots'.format(parent), folder_name)
        os.mkdir(folder)
########################################################################################################################
########################################################################################################################
#A4
def A4():
    # logs
    if not (os.path.isdir('{}/logs'.format(parent))):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/logs/A4'.format(parent))):
        folder_name = 'A4'
        folder = os.path.join('{}/logs'.format(parent), folder_name)
        os.mkdir(folder)
    # plots
    if not (os.path.isdir('{}/plots'.format(parent))):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/plots/A4'.format(parent))):
        print('creating plot logs...')
        folder_name = 'A4'
        folder = os.path.join('{}/plots'.format(parent), folder_name)
        os.mkdir(folder)
########################################################################################################################
########################################################################################################################
#A5
def A5():
    #logs
    if not(os.path.isdir('{}/logs'.format(parent))):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/logs/A5'.format(parent))):
        folder_name = 'A5'
        folder = os.path.join('{}/logs'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/logs/A5/Dropout'.format(parent))):
        folder_name = 'Early_Stopping'
        folder = os.path.join('{}/logs/A2'.format(parent), folder_name)
        os.mkdir(folder)
    #plots
    if not(os.path.isdir('{}/plots'.format(parent))):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('{}/plots/A5'.format(parent))):
        print('creating plot logs...')
        folder_name = 'A5'
        folder = os.path.join('{}/plots'.format(parent), folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('{}/plots/A5/Dropout'.format(parent))):
        folder_name = 'Early_Stopping'
        folder = os.path.join('{}/plots/A5' .format(parent), folder_name)