from PIL import Image
import numpy
import csv
import os

from copy import deepcopy
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

# User can select whether he wants to select all images or just a smaller set of them in order not to add too much noise
# after the resizing 
bound = 50

class GTSRB(DenseDesignMatrix):
    '''
        Wrapper class for gtsrb dataset. It loads training and test set and
        saves them.
    '''
    def __init__(self, which_set, img_size=None):

        path = "${PYLEARN2_DATA_PATH}/gtsrb"
        path = serial.preprocess(path)
        if which_set == 'train':
            path = path + "/Final_Training/Images"
        else:
            path = path + "/Final_Test/Images"
        self.path = path
        self.delimiter = ';'
        self.which_set = which_set
        self.img_size = img_size

        X, y = self.load_data()

        super(GTSRB, self).__init__(X=X, y=y)

    def load_data(self):

        print "\nloading data...\n"

        try:
            if self.which_set == 'train':
                path = os.path.join(self.path, 'train_' + str(self.img_size[0]) + 'x' + str(self.img_size[0]) + '.pkl.gz')
                datasets = serial.load(filepath=path)
                X, y = datasets[0], datasets[1]
            else:
                path = os.path.join(self.path, 'test_' + str(self.img_size[0]) + 'x' + str(self.img_size[0]) + '.pkl.gz')
                datasets = serial.load(filepath=path)
                X, y = datasets[0], datasets[1]
        except:
            
            if self.which_set == 'train':
    
                first = True
    
                # loop over all 43 classes
                for c in xrange(43):
                    prefix = self.path + '/' + format(c, '05d') + '/'  # subdirectory for class
                    with open(prefix + 'GT-'+ format(c, '05d') + '.csv') as f:  # annotations file
                        reader = csv.reader(f, delimiter = self.delimiter)  # csv parser for annotations file
                        reader.next()  # skip header
                        if first:
                            X, y = self.make_matrices(reader, prefix)
                            first = False
                        else:
                            next_X, next_y = self.make_matrices(reader, prefix)
                            X = numpy.append(X, next_X, axis=0)
                            y = numpy.append(y, next_y, axis=0)
    
            else:
    
                with open(self.path + '/' + "GT-final_test.csv") as f:
                    reader = csv.reader(f, delimiter = self.delimiter)  # csv parser for annotations file
                    reader.next() # skip header
                    X, y = self.make_matrices(reader)
        
            datasets = X, y
            if self.which_set == 'train':
                path = os.path.join(self.path, 'train_' + str(self.img_size[0]) + 'x' + str(self.img_size[0]) + '.pkl.gz')
                serial.save(filepath=path, obj=datasets)
            else:
                path = os.path.join(self.path, 'test_' + str(self.img_size[0]) + 'x' + str(self.img_size[0]) + '.pkl.gz')
                serial.save(filepath=path, obj=datasets)

    def make_matrices(self, reader, prefix = None):

        first = True

        for row in reader:
            if self.which_set == 'train':
                img = Image.open(prefix + '/' + row[0])
            else:
                img = Image.open(self.path + '/' + row[0])
            # crop images to get a squared image
            if img.size[0] > img.size[1]:
                img = img.crop([0, 0, img.size[1], img.size[1]])
            elif img.size[0] < img.size[1]:
                img = img.crop([0, 0, img.size[0], img.size[0]])
            if img.size[0] + bound >= self.img_size[0]:
                img = img.resize(self.img_size, Image.ANTIALIAS)  # resize
                if first:
                    X = numpy.asarray([img.getdata()])
                    y = numpy.asarray(row[7])
                    first = False
                else:
                    X = numpy.append(X, [img.getdata()], axis = 0)
                    y = numpy.append(y, row[7])

        return X, y

    def shuffle(self, X, y):
        # shuffle
        assert X.shape[0] == y.shape[0]

        indices = numpy.arange(X.shape[0])
        rng = numpy.random.RandomState()   # if given an int argument will give reproducible results
        rng.shuffle(indices)
        # shuffle both the arrays consistently
        i = 0
        temp_X = deepcopy(X)
        temp_y = deepcopy(y)
        for idx in indices:
            X[i] = temp_X[idx]
            y[i] = temp_y[idx]
            i += 1

        return X, y

    def split_rgb(self, X):
        ''' 
            modify the matrix in such a way that each image 
            is stored with a rgb configuration (all reds, 
            all greens and all blues)
        '''

        first = True
        for img in X:
            r, g, b = img[:, 0], img[:, 1], img[:, 2]
            if first == True:
                rgb = numpy.asarray([numpy.concatenate([r, g, b])])
                first = False
            else:
                rgb = numpy.append(rgb, [numpy.concatenate([r, g, b])], axis=0)

        return rgb

    def make_one_hot(self, y):
        # build the one_hot matrix used to specify labels       
        if self.one_hot:
            one_hot = numpy.zeros((y.shape[0], 43))
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.

        return one_hot
