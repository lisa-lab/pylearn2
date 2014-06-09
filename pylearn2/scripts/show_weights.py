#!/usr/bin/env python
"""
Visualizes the weight matrices of a pickled model
"""
import argparse

from pylearn2.gui import get_weights_report


def show_weights(model_path, rescale="individual",
                 border=False, out=None):
    """
    Show or save weights to an image for a pickled model

    Parameters
    ----------
    model_path : str
        Path of the model to show weights for
    rescale : str
        WRITEME
    border : bool, optional
        WRITEME
    out : str, optional
        Output file to save weights to
    """
    pv = get_weights_report.get_weights_report(model_path=model_path,
                                               rescale=rescale,
                                               border=border)

    if out is None:
        pv.show()
    else:
        pv.save(out)


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--rescale", default="individual")
    parser.add_argument("--out", default=None)
    parser.add_argument("--border", action="store_true", default=False)
    parser.add_argument("path")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    show_weights(args.path, args.rescale, args.border, args.out)
