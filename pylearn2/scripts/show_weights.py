#!/usr/bin/env python
#usage: show_weights.py model.pkl
from pylearn2.gui import get_weights_report
from optparse import OptionParser

def main():
    parser = OptionParser()

    parser.add_option("--rescale",dest='rescale',type='string',default="individual")
    parser.add_option("--out",dest="out",type='string',default=None)
    parser.add_option("--border", dest="border", action="store_true",default=False)

    options, positional = parser.parse_args()

    assert len(positional) == 1
    path ,= positional

    rescale = options.rescale
    border = options.border

    pv = get_weights_report.get_weights_report(model_path = path, rescale = rescale, border = border)

    if options.out is None:
        pv.show()
    else:
        pv.save(options.out)

if __name__ == "__main__":
    main()
