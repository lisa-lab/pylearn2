#Utilities for working with environment variables
import os

def putenv(key, value):
    #this makes the change visible to other parts of the code
    #in this same process
    os.environ[key] = value
    # this makes it available to any subprocesses we launch
    os.putenv(key, value)

