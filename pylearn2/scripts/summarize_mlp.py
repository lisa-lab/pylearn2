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

for layer in model.layers:
    print layer.layer_name
    input_space = layer.get_input_space()
    print '\tInput space: ', input_space
    print '\tTotal input dimension: ', input_space.get_total_dimension()
