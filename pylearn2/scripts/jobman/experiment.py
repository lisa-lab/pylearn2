# Standard library imports
import argparse
import datetime
import gc
import os
import warnings

# Third-party imports
import numpy as np

# Local imports
import pylearn2.config.yaml_parse
from pylearn2.utils import serial
from pylearn2.monitor import Monitor

from jobman.tools import DD, flatten
import jobman


def experimentModel(state, channel):
    """
    This function takes the state as an input and run the code in state.extract_results 
    on the model that is composed of (state.yaml_template and hyper_parameters). To know how to
    use this function, check the example in tester.py which can be found at the same directory of this file.
    """
    yaml_template = state.yaml_template
    hyper_parameters = state.hyper_parameters
    #This will be the complete yaml file that should be executed
    final_yaml_str = yaml_template % hyper_parameters

    varname = "PYLEARN2_TRAIN_FILE_NAME"
    config_file_name = "abc"
    # this makes it available to other sections of code in this same script
    os.environ[varname] = config_file_name
    print config_file_name
    # this make it available to any subprocesses we launch
    os.putenv(varname, config_file_name)
    train_obj = pylearn2.config.yaml_parse.load(final_yaml_str)

    try:
        iter(train_obj)
        iterable = True
    except TypeError as e:
        iterable = False
    if iterable:
        print '''Current implementation does not support running multiple models in one yaml file
        Please change the yam template and parameters to contain only one single model '''
    else:
        print "Executing the model."
        train_obj.main_loop()        
        #This line will call a function defined by the user and pass train_obj to it.
        #import pdb;pdb.set_trace()
        state.results = jobman.tools.resolve(state.extract_results)(train_obj)
        return channel.COMPLETE
