#!/usr/bin/python
#usage: python show_dimred_weights <whitener network>
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
