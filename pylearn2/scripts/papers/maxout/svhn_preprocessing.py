from pylearn2.datasets import preprocessing
from pylearn2.datasets.svhn import SVHN

path = '${SVHN_LOCAL_PATH}'

# Load train data
train = SVHN('splitted_train', path = path)

# prepare preprocessing
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalizationPyTables())
pipeline.items.append(preprocessing.LeCunLCN((32,32)))

# apply preprocessing to train
train.apply_preprocessor(pipeline, can_fit = True)
del train

# load and preprocess valid
valid = SVHN('valid', path = path)
valid.apply_preprocessor(pipeline, can_fit = False)

# load and preprocess test
test = SVHN('test', path = path)
test.apply_preprocessor(pipeline, can_fit = False)


