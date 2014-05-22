#!/usr/bin/env python
"""
Script to obtain version of Python modules and basic information on the
experiment setup (e.g. cpu, os).
e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
     CPU: x86_64
     OS: Linux-2.6.35.14-106.fc14.x86_64-x86_64-with-fedora-14-Laughlin
"""
__authors__ = "Olivier Dellaleau and Raul Chandias Ferrari"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Olivier Dellaleau", "Raul Chandias Ferrari"]
__license__ = "3-clause BSD"
__maintainer__ = "Raul Chandias Ferrari"
__email__ = "chandiar@iro"


import argparse
import sys

from pylearn2.utils.track_version import LibVersion


def main():
    """
    Executable entry point.

    Returns
    -------
    rval : int
        0 on success, and a non-zero error code on failure.
    """
    args = parse_args()

    # Obtain versions of the various Python packages.
    libv = LibVersion()
    libv.print_versions()
    libv.print_exp_env_info(args.print_theano)

    return 0


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    WRITEME : WRITEME
        Parsed arguments
    """
    # The global program parser.
    parser = argparse.ArgumentParser(
            description='Obtain versions of relevant Python modules.')
    parser.add_argument('-p', '--print_theano_config', action='store_true',
                        dest='print_theano', help='''If set to true, theano config
                        will be displayed.''')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
