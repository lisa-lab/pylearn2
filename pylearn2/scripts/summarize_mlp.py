#!/usr/bin/env python
__author__ = "Ian Goodfellow"
"""
Usage: summarize_mlp.py <pickle file containing an MLP>
Prints out summary info about a saved MLP.
Feel free to add more printouts.
"""

import sys

from pylearn2.utils import serial

_, model_path = sys.argv

model = serial.load(model_path)

print model
