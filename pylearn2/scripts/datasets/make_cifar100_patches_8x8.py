"""
This script makes a dataset of two million approximately whitened patches, extracted at random uniformly
from the CIFAR-100 train dataset.

This script is intended to reproduce the preprocessing used by Adam Coates et. al. in their work from
the first half of 2011 on the CIFAR-10 and STL-10 datasets.
"""
from __future__ import print_function

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.utils import string

data_dir = string.preprocess('${PYLEARN2_DATA_PATH}')

print('Loading CIFAR-100 train dataset...')
data = CIFAR100(which_set = 'train')

print("Preparing output directory...")
patch_dir = data_dir + '/cifar100/cifar100_patches_8x8'
serial.mkdir( patch_dir )
README = open(patch_dir + '/README','w')

README.write("""
The .pkl files in this directory may be opened in python using
cPickle, pickle, or pylearn2.serial.load.

data.pkl contains a pylearn2 Dataset object defining an unlabeled
dataset of 2 million 8x8 approximately whitened, contrast-normalized
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

print("Preprocessing the data...")
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=2*1000*1000))
pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
pipeline.items.append(preprocessing.ZCA())
data.apply_preprocessor(preprocessor = pipeline, can_fit = True)

data.use_design_loc(patch_dir + '/data.npy')

serial.save(patch_dir + '/data.pkl',data)

serial.save(patch_dir + '/preprocessor.pkl',pipeline)
