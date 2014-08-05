"""
This script makes a dataset of 32x32 approximately
whitened CIFAR-10 images.
"""

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.cifar100 import CIFAR100


def create_output_dir(data_dir):
    """
    Preparation of the directory.

    Parameters
    ----------
    data_dir: str
        Path of the cifar100 directory.
    """
    output_dir = data_dir + '/whitened'
    serial.mkdir(output_dir)
    README = open(output_dir + '/README', 'w')

    README.write("""
    The .pkl files in this directory may be opened in python using
    cPickle, pickle, or pylearn2.serial.load.

    train.pkl, and test.pkl each contain
    a pylearn2 Dataset object defining a labeled
    dataset of an approximately whitened version of the CIFAR-100
    dataset. train.pkl contains labeled train examples. test.pkl
    contains labeled test examples.

    preprocessor.pkl contains a pylearn2 ZCA object that was used
    to approximately whiten the images. You may want to use this
    object later to preprocess other images.

    They were created with the pylearn2 script make_cifar10_whitened.py.

    All other files in this directory, including this README, were
    created by the same script and are necessary for the other files
    to function correctly.
    """)

    README.close()

    return output_dir


def save_dataset(output_dir, dataset, name):
    """
    Save the newly created dataset to the given directory.

    Parameters
    ----------
    output_dir: str
        Path of the directory where to save the dataset.
    dataset: pylearn2.datasets.Dataset
        The dataset to save.
    name: str
        Name of the file to save.
    """
    dataset.use_design_loc(output_dir + '/' + name + '.npy')
    serial.save(output_dir + '/' + name + '.pkl', dataset)

if __name__ == '__main__':
    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar100')

    print 'Loading CIFAR-100 train dataset...'
    train = CIFAR100(which_set='train')

    print "Preparing output directory..."
    output_dir = create_output_dir(data_dir)

    print ("Learning the preprocessor and preprocessing"
           "the unsupervised train data...")
    preprocessor = preprocessing.ZCA()
    train.apply_preprocessor(preprocessor=preprocessor, can_fit=True)

    print 'Saving the unsupervised data'
    save_dataset(output_dir, train, 'train')

    print "Loading the test data"
    test = CIFAR100(which_set='test')

    print "Preprocessing the test data"
    test.apply_preprocessor(preprocessor=preprocessor, can_fit=False)

    print "Saving the test data"
    save_dataset(output_dir, test, 'test')

    serial.save(output_dir + '/preprocessor.pkl', preprocessor)
