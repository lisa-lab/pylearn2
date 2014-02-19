"""
Test for softmax_regression.ipynb
"""

import os

import pylearn2
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse


def test():
    skip_if_no_data()
    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
              'sr_dataset.yaml'), 'r') as f:
        dataset = f.read()
    f.close()
    hyper_params = {'train_stop': 50}
    dataset = dataset % (hyper_params)

    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
                           'sr_model.yaml'), 'r') as f:
        model = f.read()
    f.close()

    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
                           'sr_algorithm.yaml'), 'r') as f:
        algorithm = f.read()
    f.close()
    hyper_params = {'batch_size': 10,
                    'valid_stop': 50050}
    algorithm = algorithm % (hyper_params)

    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
                           'sr_train.yaml'), 'r') as f:
        train = f.read()
    f.close()
    train = train % locals()

    train = yaml_parse.load(train)
    train.main_loop()

if __name__ == '__main__':
    test()
