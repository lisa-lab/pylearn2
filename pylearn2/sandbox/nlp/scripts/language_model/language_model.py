#!/usr/bin/env/python
"""
A very simple language model in pylearn2
"""
import os

import pylearn2

from pylearn2.scripts.tests.yaml_testing import yaml_file_execution

yaml_file_execution(
    os.path.join(pylearn2.__path__[0],
                 "sandbox/nlp/scripts/language_model/language_model.yaml")
)
