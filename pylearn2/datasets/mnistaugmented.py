import numpy
import csv
import cPickle

from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.augment_input import augment_input

'''
    Loads MNIST dataset and build augmented dataset
    for DBM discriminative finetuning
'''

class MNISTAUGMENTED(DenseDesignMatrix):
    
    def __init__(self, which_set, model, mf_steps, one_hot = True,
                 start = None, stop = None):
        
        first_path = "${PYLEARN2_DATA_PATH}/mnistaugmented"
        first_path = serial.preprocess(first_path)
        if which_set == 'train':
            path = first_path + "/digitstrain.csv"
        else:
            path = first_path + "/digitstest.csv"
        self.path = path
        self.which_set = which_set
        self.delimiter = ','
        self.one_hot = one_hot
        self.start = start
        self.stop = stop
        self.model = model
        
        try:
            if which_set == 'train':
                datasets = load_from_dump(dump_data_dir = first_path, dump_filename = 'train_dump.pkl.gz')
                augmented_X, y = datasets[0], datasets[1]
            else:
                datasets = load_from_dump(dump_data_dir = first_path, dump_filename = 'test_dump.pkl.gz')
                augmented_X, y = datasets[0], datasets[1]
        except:
            try:
                if which_set == 'train':
                    datasets = load_from_dump(dump_data_dir = first_path, dump_filename = 'noaug_train_dump.pkl.gz')
                    X, y = datasets[0], datasets[1]
                else:
                    datasets = load_from_dump(dump_data_dir = first_path, dump_filename = 'noaug_test_dump.pkl.gz')
                    X, y = datasets[0], datasets[1]
            
            except:
                X, y = self.load_data()
                print "\ndata loaded!\n"
                
                noaug_datasets = X, y # not augmented datasets is saved in order not to waste time reloading mnist each time
                if which_set == 'train':
                    save_to_dump(var_to_dump = noaug_datasets, dump_data_dir = first_path, dump_filename = 'noaug_train_dump.pkl.gz')
                else:
                    save_to_dump(var_to_dump = noaug_datasets, dump_data_dir = first_path, dump_filename = 'noaug_test_dump.pkl.gz')

            X, y = X.astype(float), y.astype(float)
            X /= 255.    
        
            # BUILD AUGMENTED INPUT FOR FINETUNING
            augmented_X = augment_input(X, model, mf_steps)
            
            datasets = augmented_X, y
            if which_set == 'train':
                save_to_dump(var_to_dump = datasets, dump_data_dir = first_path, dump_filename = 'train_dump.pkl.gz')
            else:
                save_to_dump(var_to_dump = datasets, dump_data_dir = first_path, dump_filename = 'test_dump.pkl.gz')
        
        augmented_X, y = augmented_X[self.start:self.stop], y[self.start:self.stop]
        super(MNISTAUGMENTED, self).__init__(X = augmented_X, y = y)
        
    def load_data(self):
        
        print "\nloading data...\n"

        gtFile = open(self.path)
        reader = csv.reader(gtFile, delimiter = self.delimiter)
        
        first = True
        for row in reader:
            if first:
                X = numpy.asarray([row])
                row = reader.next()
                y = numpy.asarray([row])
                first = False
            else:
                X = numpy.append(X, [row], axis = 0)
                row = reader.next()
                y = numpy.append(y, [row], axis = 0)

        return X, y

def load_from_dump(dump_data_dir, dump_filename):
    load_file = open(dump_data_dir + "/" + dump_filename)
    unpickled_var = cPickle.load(load_file)
    load_file.close()
    return unpickled_var

def save_to_dump(var_to_dump, dump_data_dir, dump_filename):
    save_file = open(dump_data_dir + "/" + dump_filename, 'wb')  # this will overwrite current contents
    cPickle.dump(var_to_dump, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()