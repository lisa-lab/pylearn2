#!/usr/bin/env python
"""
Script to obtain version of Python modules and basic information on the
experiment setup (e.g. cpu, os), e.g.

* numpy: 1.6.1 | pylearn: a6e634b83d | pylearn2: 57a156beb0
* CPU: x86_64
* OS: Linux-2.6.35.14-106.fc14.x86_64-x86_64-with-fedora-14-Laughlin

You can also define the modules to be tracked with the environment
variable `PYLEARN2_TRACK_MODULES`.  Use ":" to separate module names
between them, e.g. `PYLEARN2_TRACK_MODULES = module1:module2:module3`

By default, the following modules are tracked: pylearn2, theano, numpy, scipy
"""
__authors__ = "Olivier Dellaleau and Raul Chandias Ferrari"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Olivier Dellaleau", "Raul Chandias Ferrari"]
__license__ = "3-clause BSD"
__maintainer__ = "Raul Chandias Ferrari"
__email__ = "chandiar@iro"


import copy
import logging
import os
import platform
import socket
import subprocess
import sys
import warnings

from theano.compat import six

logger = logging.getLogger(__name__)


class MetaLibVersion(type):
    """
    Constructor that will be called everytime another's class
    constructor is called (if the "__metaclass__ = MetaLibVersion"
    line is present in the other class definition).

    Parameters
    ----------
    cls : WRITEME
    name : WRITEME
    bases : WRITEME
    dict : WRITEME
    """

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        cls.libv = LibVersion()


class LibVersion(object):
    """
    Initialize a LibVersion object that will store the version of python
    packages in a dictionary (versions).  The python packages that are
    supported are: pylearn, pylearn2, theano, jobman, numpy and scipy.

    The key for the versions dict is the name of the package and the
    associated value is the version number.
    """

    def __init__(self):
        self.versions = {}
        self.str_versions = ''
        self.exp_env_info = {}
        self._get_lib_versions()
        self._get_exp_env_info()

    def _get_exp_env_info(self):
        """
        Get information about the experimental environment such as the
        cpu, os and the hostname of the machine on which the experiment
        is running.
        """
        self.exp_env_info['host'] = socket.gethostname()
        self.exp_env_info['cpu'] = platform.processor()
        self.exp_env_info['os'] = platform.platform()
        if 'theano' in sys.modules:
            self.exp_env_info['theano_config'] = sys.modules['theano'].config
        else:
            self.exp_env_info['theano_config'] = None

    def _get_lib_versions(self):
        """Get version of Python packages."""
        repos = os.getenv('PYLEARN2_TRACK_MODULES', '')
        default_repos = 'pylearn2:theano:numpy:scipy'
        repos = default_repos + ":" + repos
        repos = set(repos.split(':'))
        for repo in repos:
            try:
                if repo == '':
                    continue
                __import__(repo)
                if hasattr(sys.modules[repo], '__version__'):
                    v = sys.modules[repo].__version__
                    if v != 'unknown':
                        self.versions[repo] = v
                        continue
                self.versions[repo] = self._get_git_version(
                    self._get_module_parent_path(sys.modules[repo]))
            except ImportError:
                self.versions[repo] = None

        known = copy.copy(self.versions)
        # Put together all modules with unknown versions.
        unknown = [k for k, w in known.items() if not w]
        known = dict((k, w) for k, w in known.items() if w)

        # Print versions.
        self.str_versions = ' | '.join(
            ['%s:%s' % (k, w) for k, w in sorted(six.iteritems(known))] +
            ['%s:?' % ','.join(sorted(unknown))])

    def __str__(self):
        """
        Return version of the Python packages as a string.
        e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
        """
        return self.str_versions

    def _get_git_version(self, root):
        """
        Return the git revision of a repository with the letter 'M'
        appended to the revision if the repo was modified.

        e.g. 10d3046e85 M

        Parameters
        ----------
        root : str
            Root folder of the repository

        Returns
        -------
        rval : str or None
            A string with the revision hash, or None if it could not be
            retrieved (e.g. if it is not actually a git repository)
        """
        if not os.path.isdir(os.path.join(root, '.git')):
            return None
        cwd_backup = os.getcwd()
        try:
            os.chdir(root)
            sub_p = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            version = sub_p.communicate()[0][0:10].strip()
            sub_p = subprocess.Popen(['git', 'diff', '--name-only'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            modified = sub_p.communicate()[0]
            if len(modified):
                version += ' M'
            return version
        except Exception:
            pass
        finally:
            try:
                os.chdir(cwd_backup)
            except Exception:
                warnings.warn("Could not chdir back to " + cwd_backup)

    def _get_hg_version(self, root):
        """Same as `get_git_version` but for a Mercurial repository."""
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
        """Return path to a given module."""
        return os.path.realpath(module.__path__[0])

    def _get_module_parent_path(self, module):
        """Return path to the parent directory of a given module."""
        return os.path.dirname(self._get_module_path(module))

    def print_versions(self):
        """
        Print version of the Python packages as a string.
        e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
        """
        logger.info(self.__str__())

    def print_exp_env_info(self, print_theano_config=False):
        """
        Return basic information about the experiment setup such as
        the hostname of the machine the experiment was run on, the
        operating system installed on the machine.

        Parameters
        ----------
        print_theano_config : bool, optional
            If True, information about the theano configuration will be
            displayed.
        """
        logger.info('HOST: {0}'.format(self.exp_env_info['host']))
        logger.info('CPU: {0}'.format(self.exp_env_info['cpu']))
        logger.info('OS: {0}'.format(self.exp_env_info['os']))
        if print_theano_config:
            logger.info(self.exp_env_info['theano_config'])
