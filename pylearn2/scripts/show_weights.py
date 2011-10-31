#!/usr/bin/python
#usage: show_weights.py model.pkl
from pylearn2.gui import get_weights_report
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--rescale",dest='rescale',type='string',default="individual")
parser.add_option("--out",dest="out",type='string',default=None)

options, positional = parser.parse_args()

assert len(positional) == 1
path ,= positional

rescale = options.rescale

pv = get_weights_report.get_weights_report(path, rescale)

if options.out is None:
    pv.show()
else:
    pv.save(options.out)
