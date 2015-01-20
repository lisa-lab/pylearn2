"""
This script makes a dataset of two million approximately whitened patches, extracted at random uniformly
from a downsampled version of the STL-10 unlabeled and train dataset.

It assumes that you have already run make_downsampled_stl10.py, which downsamples the STL-10 images to
1/3 of their original resolution.

This script is intended to reproduce the preprocessing used by Adam Coates et. al. in their work from
the first half of 2011.
"""
from __future__ import print_function

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils as string
import numpy as np

data_dir = string.preprocess('${PYLEARN2_DATA_PATH}/stl10')

print('Loading STL10-10 unlabeled and train datasets...')
downsampled_dir = data_dir + '/stl10_32x32'

data = serial.load(downsampled_dir + '/unlabeled.pkl')
supplement = serial.load(downsampled_dir + '/train.pkl')

print('Concatenating datasets...')
data.set_design_matrix(np.concatenate((data.X,supplement.X),axis=0))
del supplement


print("Preparing output directory...")
patch_dir = data_dir + '/stl10_patches'
serial.mkdir( patch_dir )
README = open(patch_dir + '/README','w')

README.write("""
The .pkl files in this directory may be opened in python using
cPickle, pickle, or pylearn2.serial.load.

data.pkl contains a pylearn2 Dataset object defining an unlabeled
dataset of 2 million 6x6 approximately whitened, contrast-normalized
patches drawn uniformly at random from a downsampled (to 32x32)
version of the STL-10 train and unlabeled datasets.

preprocessor.pkl contains a pylearn2 Pipeline object that was used
to extract the patches and approximately whiten / contrast normalize
them. This object is necessary when extracting features for
supervised learning or test set classification, because the
extracted features must be computed using inputs that have been
whitened with the ZCA matrix learned and stored by this Pipeline.

They were created with the pylearn2 script make_stl10_patches.py.

All other files in this directory, including this README, were
created by the same script and are necessary for the other files
to function correctly.
""")

README.close()

print("Preprocessing the data...")
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(6,6),num_patches=2*1000*1000))
pipeline.items.append(preprocessing.GlobalContrastNormalization(use_std=True, sqrt_bias=10.))
pipeline.items.append(preprocessing.ZCA())
data.apply_preprocessor(preprocessor = pipeline, can_fit = True)

data.use_design_loc(patch_dir + '/data.npy')

serial.save(patch_dir + '/data.pkl',data)

serial.save(patch_dir + '/preprocessor.pkl',pipeline)
