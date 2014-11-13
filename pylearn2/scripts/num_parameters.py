#!/usr/bin/env python
"""
Usage: python num_parameters.py <model_file>.pkl

Prints the number of parameters in a saved model (total number of scalar
elements in all the arrays parameterizing the model).
"""
from __future__ import print_function

__author__ = "Ian Goodfellow"

import sys

from pylearn2.utils import serial


def num_parameters(model):
    """
    .. todo::

        WRITEME
    """
    params = model.get_params()
    return sum(map(lambda x: x.get_value().size, params))

if __name__ == '__main__':
    _, model_path = sys.argv
    model = serial.load(model_path)
    print(num_parameters(model))
