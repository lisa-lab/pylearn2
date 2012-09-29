import pylearn2
import os
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"


def list_files(suffix=""):
    """ Lists all files in pylearn2 whose filepath
    ends with suffix """

    pl2_path, = pylearn2.__path__

    return _list_files(pl2_path, suffix)


def _list_files(path, suffix=""):
    if os.path.isdir(path):
        incomplete = os.listdir(path)
        complete = [os.path.join(path, entry) for entry in incomplete]
        lists = [_list_files(subpath, suffix) for subpath in complete]
        flattened = []
        for l in lists:
            for elem in l:
                flattened.append(elem)
        return flattened
    else:
        assert os.path.exists(path)
        if path.endswith(suffix):
            return [path]
        return []

if __name__ == '__main__':
    result = list_files('.py')
    for path in result:
        print path
