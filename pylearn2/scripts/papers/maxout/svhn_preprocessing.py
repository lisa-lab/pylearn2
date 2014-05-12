import os
import logging
import shutil
import tempfile
from theano import config
from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN
from pylearn2.utils.string_utils import preprocess

orig_path = preprocess('${PYLEARN2_DATA_PATH}/SVHN/format2/h5/')
try:
    local_path = preprocess('${PYLEARN2_LOCAL_DATA_PATH}/gcn_lcn/h5/')
except ValueError:
    raise ValueError("You need to define PYLEARN2_LOCAL_DATA_PATH environment "
                     "variable.")

train_name = 'splitted_train_32x32.h5'
valid_name = 'valid_32x32.h5'
test_name = 'test_32x32.h5'


#--- copy data to a temp directory
tmp_path = tempfile.mkdtemp()
os.mkdir(tmp_path + "/h5")
for d_set in [train_name, valid_name, test_name]:
    src = os.path.join(orig_path, d_set)
    dest = os.path.join(tmp_path, "h5", d_set)
    logging.info("Copying data from {0} to {1}".format(src, dest))
    shutil.copyfile(src, dest)


def check_dtype(data):
    if str(data.X.dtype) != config.floatX:
        logging.warning("The dataset is saved as {}, changing theano's "
                        "floatX to the same dtype".format(data.X.dtype))
        config.floatX = str(data.X.dtype)

#--- preprocessing
# Load train data
train = SVHN('splitted_train',
             path=tmp_path,
             write_permission=True,
             enable_cache=False)
check_dtype(train)

# prepare preprocessing
pipeline = preprocessing.Pipeline()
# without batch_size there is a high chance that you might encounter
# memory error or pytables crashes
pipeline.items.append(
    preprocessing.GlobalContrastNormalization(batch_size=5000))
pipeline.items.append(preprocessing.LeCunLCN((32, 32)))

# apply the preprocessings to train
train.apply_preprocessor(pipeline, can_fit=True)
del train

# load and preprocess valid
valid = SVHN('valid', path=tmp_path, write_permission=True,
             enable_cache=False)
check_dtype(valid)
valid.apply_preprocessor(pipeline, can_fit=False)

# load and preprocess test
test = SVHN('test', path=tmp_path, write_permission=True, enable_cache=False)
check_dtype(test)
test.apply_preprocessor(pipeline, can_fit=False)


#--- copy data to local path
if not os.path.isdir(local_path):
    os.makedirs(local_path)

for d_set in [train_name, valid_name, test_name]:
    if not os.path.isfile(os.path.join(local_path, d_set)):
        src = os.path.join(tmp_path, "h5", d_set)
        dest = os.path.join(local_path, d_set)
        logging.info("Copying data from {0} to {1}".format(src, dest))
        shutil.copyfile(src, dest)
        os.remove(src)
    else:
        raise ValueError("The file {} already exists."
                         "Pleae delete it to be able to run this script")

shutil.rmtree(tmp_path)
