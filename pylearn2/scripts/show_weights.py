#!/usr/bin/env python
"""
Visualizes the weight matrices of a pickled model
"""
import argparse

from pylearn2.gui import get_weights_report


def main():
    """
    Executes this script
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--rescale", default="individual")
    parser.add_argument("--out", default=None)
    parser.add_argument("--border", action="store_true", default=False)
    parser.add_argument("path")

    options = parser.parse_args()

    pv = get_weights_report.get_weights_report(model_path=options.path,
                                               rescale=options.rescale,
                                               border=options.border)

    if options.out is None:
        pv.show()
    else:
        pv.save(options.out)


if __name__ == "__main__":
    main()
