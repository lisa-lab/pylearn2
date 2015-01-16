"""
Wrapper for the Adult UCI dataset:
http://archive.ics.uci.edu/ml/datasets/Adult
"""
__author__ = "Ian Goodfellow"

import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.utils.string_utils import preprocess


def adult(which_set):
    """
    Parameters
    ----------
    which_set : str
        'train' or 'test'

    Returns
    -------
    adult : DenseDesignMatrix
        Contains the Adult dataset.

    Notes
    -----
    This discards all examples with missing features. It would be trivial
    to modify this code to not do so, provided with a convention for how to
    treat the missing features.
    Categorical values are converted into a one-hot code.
    """

    base_path = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"), "adult")

    set_file = {'train': 'adult.data', 'test': 'adult.test'}[which_set]

    full_path = os.path.join(base_path, set_file)

    content = open(full_path, 'r').readlines()

    # strip off stupid header line they added to only the test set
    if which_set == 'test':
        content = content[1:]
    # strip off empty final line
    content = content[:-1]

    # verify # of examples
    num_examples = {'train': 32561, 'test': 16281}[which_set]
    assert len(content) == num_examples, (len(content), num_examples)

    # strip out examples with missing features, verify number of remaining
    # examples
    content = [line for line in content if line.find('?') == -1]
    num_examples = {'train': 30162, 'test': 15060}[which_set]
    assert len(content) == num_examples, (len(content), num_examples)

    # strip off endlines, separate entries
    content = map(lambda l: l[:-1].split(', '), content)

    # split data into features and targets
    features = map(lambda l: l[:-1], content)
    targets = map(lambda l: l[-1], content)
    del content

    # convert targets to binary
    assert all(map(lambda l: l in ['>50K', '<=50K', '>50K.', '<=50K.'],
                   targets))
    y = map(lambda l: [l[0] == '>'], targets)
    y = np.array(y)
    del targets

    # Process features into a design matrix
    variables = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week',
                 'native-country']
    continuous = set(['age', 'fnlwgt', 'education-num', 'capital-gain',
                      'capital-loss', 'hours-per-week'])
    assert all(var in variables for var in continuous)
    assert all(map(lambda l: len(l) == len(variables), features))
    pieces = []

    for i, var in enumerate(variables):
        data = map(lambda l: l[i], features)
        assert len(data) == num_examples
        if var in continuous:
            data = map(lambda l: float(l), data)
            data = np.array(data)
            data = data.reshape(data.shape[0], 1)
        else:
            unique_values = list(set(data))
            data = map(lambda l: unique_values.index(l), data)
            data = convert_to_one_hot(data)
        pieces.append(data)

    X = np.concatenate(pieces, axis=1)

    return DenseDesignMatrix(X=X, y=y)
