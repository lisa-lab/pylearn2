from PIL import Image
import numpy
import csv
import os

from copy import deepcopy
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.scripts.dbm.augment_input import augment_input
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing

# User can select whether he wants to select all images or just a smaller set of them in order not to add too much noise
# after the resizing 
bound = 50

class GTSRB(DenseDesignMatrix):
    '''
        Wrapper class for gtsrb dataset. It loads training and test set and
        saves them. When the augmentation set is needed, it is created and
        saved. Augmented dataset is saved only if save_aug flag is True. If
        the datasets already exist, they are loaded from the pkl files.
    '''
    def __init__(self, which_set, model=None, mf_steps=None, one_hot=True,
                 start=None, stop=None, img_size=None, save_aug=False):

        path = "${PYLEARN2_DATA_PATH}/gtsrb"
        path = serial.preprocess(path)
        if which_set == 'train':
            path = path + "/Final_Training/Images"
        else:
            path = path + "/Final_Test/Images"
        self.path = path
        self.delimiter = ';'
        self.one_hot = one_hot
        self.which_set = which_set
        self.img_size = img_size

        try:
            # check the presence of saved augmented datasets
            if which_set == 'train':
                path = os.path.join(self.path, 'aug_train_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                aug_datasets = serial.load(filepath=path)
                augmented_X, y = aug_datasets[0], aug_datasets[1]
            else:
                path = os.path.join(self.path, 'aug_test_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                aug_datasets = serial.load(filepath=path)
                augmented_X, y = aug_datasets[0], aug_datasets[1]

            X, y = augmented_X[start:stop], y[start:stop]
        except:
            # if there're not saved augmented datasets, if there're saved 
            # normal datasets, it loads and augment them, otherwise it creates
            # and augment them
            try:
                if which_set == 'train':
                    path = os.path.join(self.path, 'train_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                    datasets = serial.load(filepath=path)
                    X, y = datasets[0], datasets[1]
                else:
                    path = os.path.join(self.path, 'test_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                    datasets = serial.load(filepath=path)
                    X, y = datasets[0], datasets[1]

            except:
                X, y = self.load_data()
                print "\ndata loaded!\n"

                datasets = X, y  # not augmented datasets is saved in order not to waste time reloading gtsrb each time
                if which_set == 'train':
                    path = os.path.join(self.path, 'train_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                    serial.save(filepath=path, obj=datasets)
                else:
                    path = os.path.join(self.path, 'test_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                    serial.save(filepath=path, obj=datasets)

            X, y = X[start:stop], y[start:stop]

            # BUILD AUGMENTED INPUT FOR FINETUNING
            if mf_steps is not None:
                augmented_X = augment_input(X, model, mf_steps)

                aug_datasets = augmented_X, y
                if save_aug == True:
                    if which_set == 'train':
                        path = os.path.join(self.path, 'aug_train_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                        serial.save(filepath=path, obj=aug_datasets)
                    else:
                        path = os.path.join(self.path, 'aug_test_dump_' + str(img_size[0]) + 'x' + str(img_size[0]) + '.pkl.gz')
                        serial.save(filepath=path, obj=aug_datasets)

                X = augmented_X

        super(GTSRB, self).__init__(X=X, y=y)

    def load_data(self):

        print "\nloading data...\n"

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

        # preprocess loaded images that, until this point, have only been
        # cropped and resized
        X, y = self.preprocess(X, y)
        #print '\n' + str(bad_images) + ' images have been discarded for not respecting size requirements\n'
        return X, y

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

    def preprocess(self, X, y):

        X = self.split_rgb(X)
        if self.which_set == 'train':            
            X, y = self.shuffle(X, y)    
        y = self.make_one_hot(y)
        
        # image preprocessing
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(
            preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000))
        pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
        X.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        y.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        
        return X, y