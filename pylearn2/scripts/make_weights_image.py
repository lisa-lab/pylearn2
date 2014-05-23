#!/usr/bin/env python
"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import sys
from pylearn2.gui import get_weights_report
import warnings

warnings.warn("make_weights_image.py is deprecated. Use show_weights.py with"
        " the --out flag. make_weights_image.py may be removed on or after "
        "2014-08-28.")

if __name__ == "__main__":
    print 'loading model'
    path = sys.argv[1]
    print 'loading done'

    rescale = True

    if len(sys.argv) > 2:
        rescale = eval(sys.argv[2])

    pv = get_weights_report.get_weights_report(path, rescale)

    pv.save(sys.argv[1]+'.png')
