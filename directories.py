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
    if not(os.path.isdir('%s/logs' % parent)):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('%s/logs/A2' %parent)):
        folder_name = 'A2'
        folder = os.path.join('%s/logs' % parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('%s/logs/A2/Single_Layer' % parent)):
        folder_name = 'Single_Layer'
        folder = os.path.join('%s/logs/A2' % parent, folder_name)
        os.mkdir(folder)
    #plots
    if not(os.path.isdir('%s/plots' % parent)):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('%s/plots/A2' %parent)):
        print('creating plot logs...')
        folder_name = 'A2'
        folder = os.path.join('%s/plots' % parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('%s/logs/A2/Single_Layer' % parent)):
        folder_name = 'Single_Layer'
        folder = os.path.join('%s/plots/A2' % parent, folder_name)
        os.mkdir(folder)
#######################################################################################################################
def extra_layer():
    #logs
    if not(os.path.isdir('%s/logs' % parent)):
        print('creating logs...')
        folder_name = 'logs'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('%s/logs/A2' %parent)):
        folder_name = 'A2'
        folder = os.path.join('%s/logs' % parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('%s/logs/A2/Extra_Layer' % parent)):
        folder_name = 'Extra_Layer'
        folder = os.path.join('%s/logs/A2' % parent, folder_name)
        os.mkdir(folder)
    #plots
    if not(os.path.isdir('%s/plots' % parent)):
        folder_name = 'plots'
        folder = os.path.join(parent, folder_name)
        os.mkdir(folder)
    if not(os.path.isdir('%s/plots/A2' %parent)):
        print('creating plot logs...')
        folder_name = 'A2'
        folder = os.path.join('%s/plots' % parent, folder_name)
        os.mkdir(folder)
    if not (os.path.isdir('%s/logs/A2/Extra_Layer' % parent)):
        folder_name = 'Extra_Layer'
        folder = os.path.join('%s/plots/A2' % parent, folder_name)
        os.mkdir(folder)