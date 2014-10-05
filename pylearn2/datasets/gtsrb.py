from PIL import Image
import numpy
import csv
import cPickle

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.augment_input import augment_input

# User can select whether he wants to select all images or just a smaller set of them in order not to add too much noise
# after the resizing 
bound_train = 1000
bound_test = 1000

class GTSRB(DenseDesignMatrix):
    
    def __init__(self, which_set, model = None, mf_steps = None, one_hot = True,
                 start = None, stop = None, img_size = None):
        
        #path = "${PYLEARN2_DATA_PATH}/gtsrb"
        first_path = "/home/deramo/workspace/datasets/gtsrb"
        if which_set == 'train':
            path = first_path + "/Final_Training/Images"
        else:
            path = first_path + "/Final_Test/Images"
        self.path = path
        self.which_set = which_set
        self.delimiter = ';'
        self.img_size = img_size
        self.one_hot = one_hot
        self.start = start
        self.stop = stop
        
        try:
            if which_set == 'train':
                datasets = load_from_dump(dump_data_dir = self.path, dump_filename = 'train_dump.pkl.gz')
                augmented_X, y = datasets[0], datasets[1]
            else:
                datasets = load_from_dump(dump_data_dir = self.path, dump_filename = 'test_dump.pkl.gz')
                augmented_X, y = datasets[0], datasets[1]
        except:
            try:
                if which_set == 'train':
                    datasets = load_from_dump(dump_data_dir = self.path, dump_filename = 'noaug_train_dump.pkl.gz')
                    X, y = datasets[0], datasets[1]
                else:
                    datasets = load_from_dump(dump_data_dir = self.path, dump_filename = 'noaug_test_dump.pkl.gz')
                    X, y = datasets[0], datasets[1]
                    X, y = X[0:12600], y[0:12600] # temporaneo
            
            except:
                X, y = self.load_data()
                print "\ndata loaded!\n"
                
                noaug_datasets = X, y # not augmented datasets is saved in order not to waste time reloading gtsrb each time
                if which_set == 'train':
                    save_to_dump(var_to_dump = noaug_datasets, dump_data_dir = self.path, dump_filename = 'noaug_train_dump.pkl.gz')
                else:
                    save_to_dump(var_to_dump = noaug_datasets, dump_data_dir = self.path, dump_filename = 'noaug_test_dump.pkl.gz')
            
            X, y = X.astype(float), y.astype(float)
            X /= 255.
            
            # BUILD AUGMENTED INPUT FOR FINETUNING
            if mf_steps is not None:
                augmented_X = augment_input(X, model, mf_steps)
                
                datasets = augmented_X, y
                if which_set == 'train':
                    save_to_dump(var_to_dump = datasets, dump_data_dir = self.path, dump_filename = 'train_dump.pkl.gz')
                else:
                    save_to_dump(var_to_dump = datasets, dump_data_dir = self.path, dump_filename = 'test_dump.pkl.gz')
        
        super(GTSRB, self).__init__(X = X, y = y)
        
    def load_data(self):
        
        print "\nloading data...\n"

        if self.which_set == 'train':
        
            first = True
            
            # loop over all 43 classes
            for c in xrange(43): #43
                prefix = self.path + '/' + format(c, '05d') + '/' # subdirectory for class
                f = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
                reader = csv.reader(f, delimiter = self.delimiter) # csv parser for annotations file
                reader.next() # skip header
                for row in reader:
                    img = Image.open(prefix + '/' + row[0])
                    if img.size[0] + bound_train >= self.img_size[0]:
                        img = img.convert('L')  # grayscale
                        img = img.resize(self.img_size, Image.ANTIALIAS) #resize
                        if first:
                            X = numpy.asarray([img.getdata()])
                            y = numpy.asarray(row[7])
                            first = False
                        else:
                            X = numpy.append(X, [img.getdata()], axis = 0)
                            y = numpy.append(y, row[7])
                f.close()
                
            # shuffle
            assert X.shape[0] == y.shape[0]
            
            indices = numpy.arange(X.shape[0])
            rng = numpy.random.RandomState()  # if given an int argument will give reproducible results
            rng.shuffle(indices)
            # shuffle both the arrays consistently
            i = 0
            temp_X = X
            temp_y = y
            for idx in indices:
                X[i] = temp_X[idx]
                y[i] = temp_y[idx]
                i += 1
        
        else:
            
            first = True
            
            f = open(self.path + '/' + "GT-final_test.csv")
            reader = csv.reader(f, delimiter = self.delimiter) # csv parser for annotations file
            reader.next() # skip header
            
            for c in xrange(12630):
                for row in reader:
                    img = Image.open(self.path + '/' + row[0])
                    if img.size[0] + bound_train >= self.img_size[0]:
                        img = img.convert('L')  # grayscale
                        img = img.resize(self.img_size, Image.ANTIALIAS) #resize
                        if first:
                            X = numpy.asarray([img.getdata()])
                            y = row[7]
                            first = False
                        else:
                            X = numpy.append(X, [img.getdata()], axis = 0)
                            y = numpy.append(y, row[7])
            f.close()
        
        # build the one_hot matrix used to specify labels       
        if self.one_hot:
            one_hot = numpy.zeros((y.shape[0], 43))
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot
        
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