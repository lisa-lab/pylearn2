#!/usr/bin/env python
"""
Usage: print_model.py <pickle file containing a model>
Prints out a saved model.
"""
from __future__ import print_function

__author__ = "Ian Goodfellow"

import sys

from pylearn2.utils import serial

if __name__ == "__main__":
    _, model_path = sys.argv

    model = serial.load(model_path)

    print(model)
