#!/usr/bin/env python
__authors__ = "Olivier Dellaleau and Raul Chandias Ferrari"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Olivier Dellaleau", "Raul Chandias Ferrari"]
__license__ = "3-clause BSD"
__maintainer__ = "Raul Chandias Ferrari"
__email__ = "chandiar@iro"


"""
Script to obtain version of Python modules as a string.
e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
"""


import argparse
import copy
import datetime
import os
import socket
import subprocess
import sys
import time

import jobman
import numpy
import pylearn
import pylearn2
import scipy
import theano


class LibVersion(object):
    def __init__(self):
        """
        Initialize a LibVersion object that will store the version of python
        packages in a dictionary (versions).  The python packages that are 
        supported are: pylearn, pylearn2, theano, jobman, numpy and scipy.

        The key for the versions dict is the name of the package and the
        associated value is the version number.
        """
        self.versions = {}
        self.str_versions = ''
        self._get_lib_versions()

    def _get_lib_versions(self):
        """
        Get version of Python packages.
        """
        # pylearn.
        self.versions['pylearn'] = self._get_hg_version(self._get_module_parent_path(pylearn))

        # pylearn2.
        self.versions['pylearn2'] = self._get_git_version(self._get_module_parent_path(pylearn2))

        # Theano.
        v = theano.__version__
        if v == 'unknown':
            v = self._get_git_version(self._get_module_parent_path(theano))
        self.versions['theano'] = v

        # Jobman: will only work with old assembla version (there is no version
        # number currently available when running setup.py).
        self.versions['jobman'] = self._get_hg_version(self._get_module_parent_path(jobman))

        # Numpy.
        self.versions['numpy'] = numpy.__version__

        # Scipy.
        self.versions['scipy'] = scipy.__version__
        known = copy.copy(self.versions)
        # Put together all modules with unknown versions.
        unknown = []
        for k, v in known.items():
            if v is None:
                unknown.append(k)
                del known[k]

        # Print versions.
        self.str_versions = ' | '.join(['%s:%s' % (k, v)
                               for k, v in sorted(known.iteritems())] +
                               ['%s:?' % ','.join(sorted(unknown))])

    def __str__(self):
        """
        Return version of the Python packages as a string.
        e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
        """
        return self.str_versions

    def _get_git_version(self, root):
        """
        Return the git revision of a repository.

        :param root: Root folder of the repository.

        :return: A string with the revision hash, or None if it could not be
        retrieved (e.g. if it is not actually a git repository).
        """
        if not os.path.isdir(os.path.join(root, '.git')):
            return None
        cwd_backup = os.getcwd()
        try:
            os.chdir(root)
            sub_p = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            return sub_p.communicate()[0][0:10].strip()
        finally:
            os.chdir(cwd_backup)


    def _get_hg_version(self, root):
        """
        Same as `get_git_version` but for a Mercurial repository.
        """
        if not os.path.isdir(os.path.join(root, '.hg')):
            return None
        cwd_backup = os.getcwd()
        try:
            os.chdir(root)
            sub_p = subprocess.Popen(['hg', 'parents'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            sub_p_output = sub_p.communicate()[0]
        finally:
            os.chdir(cwd_backup)
        first_line = sub_p_output.split('\n')[0]
        # The first line looks like:
        #   changeset:   1517:a6e634b83d88
        return first_line.split(':')[2][0:10]


    def _get_module_path(self, module):
        """
        Return path to a given module.
        """
        return os.path.realpath(module.__path__[0])


    def _get_module_parent_path(self, module):
        """
        Return path to the parent directory of a given module.
        """
        return os.path.dirname(self._get_module_path(module))

    def get_versions(self):
        """
        Return version of the Python packages as a string.
        e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
        """
        return self.__str__()


def main():
    """
    Executable entry point.

    :return: 0 on success, and a non-zero error code on failure.
    """
    args = parse_args()

    # Obtain versions of the various Python packages.
    libv = LibVersion()
    print libv.get_versions()

    return 0


def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    # The global program parser.
    parser = argparse.ArgumentParser(
            description='Obtain versions of relevant Python modules.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
