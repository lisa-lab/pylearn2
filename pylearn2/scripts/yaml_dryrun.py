#!/usr/bin/env python
"""Script to load (or just parse) a YAML file to verify it before launch."""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"


import argparse
from yaml import load as yaml_load
from pylearn2.config.yaml_parse import load, initialize


def main(args=None):
    """
    Execute the main body of the script.

    Parameters
    ----------
    args : list, optional
        Command-line arguments. If unspecified, `sys.argv[1:]` is used.
    """
    parser = argparse.ArgumentParser(description='Load a YAML file without '
                                                 'performing any training.')
    parser.add_argument('yaml_file', type=argparse.FileType('r'),
                        help='The YAML file to load.')
    parser.add_argument('-N', '--no-instantiate',
                        action='store_const', default=False, const=True,
                        help='Only verify that the YAML parses correctly '
                             'but do not attempt to instantiate the objects. '
                             'This might be used as a quick sanity check if '
                             'checking a file that requires a GPU in an '
                             'environment that lacks one (e.g. a cluster '
                             'head node)')
    args = parser.parse_args(args=args)
    name = args.yaml_file.name
    initialize()
    if args.no_instantiate:
        yaml_load(args.yaml_file)
        print "Successfully parsed %s (but objects not instantiated)." % name
    else:
        load(args.yaml_file)
        print "Successfully parsed and loaded %s." % name


if __name__ == "__main__":
    main()
