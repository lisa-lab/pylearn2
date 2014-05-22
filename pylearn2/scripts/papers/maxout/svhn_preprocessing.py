import os
import logging
import shutil
from theano import config
from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN
from pylearn2.utils.string_utils import preprocess

orig_path = preprocess('${PYLEARN2_DATA_PATH}/SVHN/format2')
try:
    local_path = preprocess('${SVHN_LOCAL_PATH}')
except ValueError:
    raise ValueError("You need to define SVHN_LOCAL_PATH environment "
                        "variable.")

train_name ='h5/splitted_train_32x32.h5'
valid_name = 'h5/valid_32x32.h5'
test_name = 'h5/test_32x32.h5'

# copy data if don't exist
if not os.path.isdir(os.path.join(local_path, 'h5')):
    os.makedirs(os.path.join(local_path, 'h5'))

for d_set in [train_name, valid_name, test_name]:
    if not os.path.isfile(os.path.join(local_path, d_set)):
        logging.info("Copying data from {0} to {1}".format(os.path.join(local_path, d_set), local_path))
        shutil.copyfile(os.path.join(orig_path, d_set),
                    os.path.join(local_path, d_set))

def check_dtype(data):
    if str(data.X.dtype) != config.floatX:
        logging.warning("The dataset is saved as {}, changing theano's floatX "\
                "to the same dtype".format(data.X.dtype))
        config.floatX = str(data.X.dtype)

# Load train data
train = SVHN('splitted_train', path=local_path)
check_dtype(train)

# prepare preprocessing
pipeline = preprocessing.Pipeline()
# without batch_size there is a high chance that you might encounter memory error
# or pytables crashes
pipeline.items.append(preprocessing.GlobalContrastNormalization(batch_size=5000))
pipeline.items.append(preprocessing.LeCunLCN((32,32)))

# apply the preprocessings to train
train.apply_preprocessor(pipeline, can_fit=True)
del train

# load and preprocess valid
valid = SVHN('valid', path=local_path)
check_dtype(valid)
valid.apply_preprocessor(pipeline, can_fit=False)

# load and preprocess test
test = SVHN('test', path=local_path)
check_dtype(test)
test.apply_preprocessor(pipeline, can_fit=False)
