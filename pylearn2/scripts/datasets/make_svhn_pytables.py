"""
Script to split the downloaded SVHN .mat files to 'test', 'valid', and 'train'
sets and save them as pytables. Ensure that you have set the PYLEARN2_DATA_PATH
environment variable and downloaded the .mat files to the path
${PYLEARN2_DATA_PATH}/SVHN/format2/
"""
import os
from pylearn2.datasets.svhn import SVHN
from pylearn2.utils.string_utils import preprocess

assert 'PYLEARN2_DATA_PATH' in os.environ, "PYLEARN2_DATA_PATH not defined"

orig_path = preprocess('${PYLEARN2_DATA_PATH}/SVHN/format2/')

# Check if MAT files have been downloaded
if not os.path.isdir(orig_path):
    raise IOError("You need to download the SVHN format2 dataset MAT files "
                  "before running this conversion script.")

# Create directory in which to save the pytables files
local_path = orig_path
if not os.path.isdir(os.path.join(local_path, 'h5')):
    os.makedirs(os.path.join(local_path, 'h5'))

print("***************************************************************\n"
      "Please ignore the warning produced during this MAT -> Pytables\n"
      "conversion for the SVHN dataset. If you are creating the\n"
      "pytables for the first time then no files are modified/over-written,\n"
      "they are simply written for the first time.\n"
      "***************************************************************\n")

test = SVHN('test', path=local_path)

valid = SVHN('valid', path=local_path)

train = SVHN('splitted_train', path=local_path)
