#!/usr/bin/env python
__author__ = "Ian Goodfellow"
"""
Usage: print_model.py <pickle file containing a model>
Prints out a saved model.
"""

import sys

from pylearn2.utils import serial

_, model_path = sys.argv

model = serial.load(model_path)

print model
