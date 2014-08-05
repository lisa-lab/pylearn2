"""
This script makes a dataset of two million approximately whitened patches,
extracted at random uniformly from the CIFAR-100 train dataset.

This script is intended to reproduce the preprocessing used by
Adam Coates et. al. in their work from the first half of 2011
on the CIFAR-10 and STL-10 datasets.
"""

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.utils import string_utils


def create_output_dir(data_dir):
    """
    Preparation of the directory.

    Parameters
    ----------
    data_dir: str
        Path of the cifar100 directory.
    """
    patch_dir = data_dir + '/cifar100/cifar100_patches'
    serial.mkdir(patch_dir)
    README = open(patch_dir + '/README', 'w')

    README.write("""
    The .pkl files in this directory may be opened in python using
    cPickle, pickle, or pylearn2.serial.load.

    data.pkl contains a pylearn2 Dataset object defining an unlabeled
    dataset of 2 million 6x6 approximately whitened, contrast-normalized
    patches drawn uniformly at random from the CIFAR-100 train set.

    preprocessor.pkl contains a pylearn2 Pipeline object that was used
    to extract the patches and approximately whiten / contrast normalize
    them. This object is necessary when extracting features for
    supervised learning or test set classification, because the
    extracted features must be computed using inputs that have been
    whitened with the ZCA matrix learned and stored by this Pipeline.

    They were created with the pylearn2 script make_cifar100_patches.py.

    All other files in this directory, including this README, were
    created by the same script and are necessary for the other files
    to function correctly.
    """)

    README.close()

    return patch_dir


def save_dataset(patch_dir, dataset, name):
    """
    Save the newly created dataset to the given directory.

    Parameters
    ----------
    patch_dir: str
        Path of the directory where to save the dataset.
    dataset: pylearn2.datasets.Dataset
        The dataset to save.
    name: str
        Name of the file to save.
    """
    dataset.use_design_loc(patch_dir + '/' + name + '.npy')
    serial.save(patch_dir + '/' + name + '.pkl', dataset)

if __name__ == '__main__':
    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')

    print 'Loading CIFAR-100 train dataset...'
    data = CIFAR100(which_set='train')

    print "Preparing output directory..."
    patch_dir = create_output_dir(data_dir)

    print "Preprocessing the data..."
    patches = preprocessing.ExtractPatches(patch_shape=(6, 6),
                                           num_patches=2*1000*1000)
    normalization = preprocessing.GlobalContrastNormalization(sqrt_bias=10.,
                                                              use_std=True)

    pipeline = preprocessing.Pipeline()
    pipeline.items.append(patches)
    pipeline.items.append(normalization)
    pipeline.items.append(preprocessing.ZCA())
    data.apply_preprocessor(preprocessor=pipeline, can_fit=True)

    save_dataset(patch_dir, data, 'data')
    serial.save(patch_dir + '/preprocessor.pkl', pipeline)
