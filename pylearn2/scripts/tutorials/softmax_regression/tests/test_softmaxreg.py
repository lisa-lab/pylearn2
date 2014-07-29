"""
Test for softmax_regression.ipynb
"""

import os

from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse
from theano import config


def test():
    skip_if_no_data()

    dirname = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

    with open(os.path.join(dirname, 'sr_dataset.yaml'), 'r') as f:
        dataset = f.read()

    if config.mode == "DEBUG_MODE":
        hyper_params = {'train_stop': 10}
    else:
        hyper_params = {'train_stop': 50}
    dataset = dataset % (hyper_params)

    with open(os.path.join(dirname, 'sr_model.yaml'), 'r') as f:
        model = f.read()

    with open(os.path.join(dirname, 'sr_algorithm.yaml'), 'r') as f:
        algorithm = f.read()

    if config.mode == "DEBUG_MODE":
        hyper_params = {'batch_size': 10,
                        'valid_stop': 50010}
    else:
        hyper_params = {'batch_size': 10,
                        'valid_stop': 50050}
    algorithm = algorithm % (hyper_params)

    with open(os.path.join(dirname, 'sr_train.yaml'), 'r') as f:
        train = f.read()

    save_path = os.path.dirname(os.path.realpath(__file__))
    train = train % locals()

    train = yaml_parse.load(train)
    train.main_loop()

    try:
        os.remove("{}/softmax_regression.pkl".format(save_path))
        os.remove("{}/softmax_regression_best.pkl".format(save_path))
    except OSError:
        pass

if __name__ == '__main__':
    test()
