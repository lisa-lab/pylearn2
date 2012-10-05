#!/usr/bin/python
#usage: python show_dimred_weights <whitener network>
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import sys
from pylearn2.gui import get_weights_report

print 'loading model'
path = sys.argv[1]
print 'loading done'

rescale = True

if len(sys.argv) > 2:
	rescale = eval(sys.argv[2])

pv = get_weights_report.get_weights_report(path, rescale)

pv.save(sys.argv[1]+'.png')
