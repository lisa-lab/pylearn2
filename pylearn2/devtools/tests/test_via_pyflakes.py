from __future__ import print_function

from pylearn2.devtools.run_pyflakes import run_pyflakes
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

def test_via_pyflakes():
    d = run_pyflakes(True)
    if len(d.keys()) != 0:
        print('Errors detected by pyflakes')
        for key in d.keys():
            print(key+':')
            for l in d[key].split('\n'):
                print('\t',l)

        raise AssertionError("You have errors detected by pyflakes")
