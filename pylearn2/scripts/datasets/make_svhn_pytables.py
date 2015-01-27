import os
import logging
import shutil
from theano import config
from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN
from pylearn2.utils.string_utils import preprocess

orig_path  = preprocess('${PYLEARN2_DATA_PATH}/SVHN/format2')

# Check if MAT files have been downloaded
if not os.path.isdir(orig_path):
    raise IOError("You need to download the SVNH format2 dataset MAT files "
                     "before running this conversion script.")

# Create directory in which to save the pytables files
local_path = orig_path
if not os.path.isdir(os.path.join(local_path, 'h5')):
    os.makedirs(os.path.join(local_path, 'h5'))

print """
      ***************************************************************
      Please ignore the warning produced during this MAT -> Pytables
      conversion for the SVNH dataset. If you are creating the
      pytables for the first time then no files are modified/over-written,
      they are simply written for the first time.
      ***************************************************************
      """

test = SVHN('test', path=local_path+'/')

valid = SVHN('valid', path=local_path)

train = SVHN('splitted_train', path=local_path)

