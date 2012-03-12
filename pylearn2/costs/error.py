import warnings

class SupervisedError(object):
    def __init__(self):
        pass
    def __call__(self):
        warnings.warn('You should implement a call method properly.')
		
class UnsupervisedError(object):
    def __init__(self):
        pass
    def __call__(self):
        warnings.warn('You should implement a call method properly.')
