import warnings


class SupervisedCost(object):
    def __init__(self):
        pass

    def __call__(self):
        warnings.warn('You should implement a call method properly.')


class UnsupervisedCost(object):
    def __init__(self):
        pass

    def __call__(self):
        warnings.warn('You should implement a call method properly.')
