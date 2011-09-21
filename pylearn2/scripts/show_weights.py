#!/usr/bin/python
#usage: python show_dimred_weights <whitener network>
import sys
from pylearn2.gui import get_weights_report

print 'loading model'
path = sys.argv[1]
print 'loading done'

rescale = 'individual'

if len(sys.argv) > 2:
    arg2 = sys.argv[2]
    assert arg2.startswith('--rescale=')
    split = arg2.split('--rescale=')
    assert len(split) == 2
    rescale = split[1]

pv = get_weights_report.get_weights_report(path, rescale)

pv.show()
