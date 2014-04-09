#!/usr/bin/env/python
"""
A very simple language model in pylearn2
"""
import os
from pylearn2.scripts.tests.yaml_testing import yaml_file_execution

yaml_file_execution(os.path.join(os.getcwd(), "language_model.yaml"))
