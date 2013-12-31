#!/usr/bin/env python
__authors__ = "Olivier Dellaleau and Raul Chandias Ferrari"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Olivier Dellaleau", "Raul Chandias Ferrari"]
__license__ = "3-clause BSD"
__maintainer__ = "Raul Chandias Ferrari"
__email__ = "chandiar@iro"


"""
Script to obtain version of Python modules and basic information on the
experiment setup (e.g. cpu, os)
e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
     CPU: x86_64
     OS: Linux-2.6.35.14-106.fc14.x86_64-x86_64-with-fedora-14-Laughlin

You can also define the modules to be tracked with the environment variable
PYLEARN2_TRACK_MODULES.  Use ":" to seperate module names between them, e.g.
PYLEARN2_TRACK_MODULES = module1:module2:module3

By default, the following modules are tracked: pylearn2, theano, numpy, scipy

"""

import copy
import importlib
import os
import platform
import socket
import subprocess
import sys


class MetaLibVersion(type):
    def __init__(cls, name, bases, dict):
        """
        Constructor that will be called everytime another's class constructor
        is called (if the "__metaclass__ = MetaLibVersion" line is present in the
        other class definition).
        """
        type.__init__(cls, name, bases, dict)
        cls.libv = LibVersion()


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
        self.exp_env_info = {}
        self._get_lib_versions()
        self._get_exp_env_info()
        
    def _get_exp_env_info(self):
	"""
	Get information about the experimental environment such as the cpu, os and
        the hostname of the machine on which the experiment is running.
	"""
	self.exp_env_info['host'] = socket.gethostname()
	self.exp_env_info['cpu'] = platform.processor()
	self.exp_env_info['os'] = platform.platform()
	if 'theano' in sys.modules:
	    self.exp_env_info['theano_config'] = sys.modules['theano'].config
	else:
	    self.exp_env_info['theano_config'] = None

    def _get_lib_versions(self):
        """
        Get version of Python packages.
        """
        repos = os.getenv('PYLEARN2_TRACK_MODULES', None)
        if repos is not None:
            repos = repos.split(':')
            for repo in repos:
	        try:
		    importlib.import_module(repo)
		    self.versions[repo] = self._get_git_version(self._get_module_parent_path(sys.modules[repo]))
	        except ImportError:
		    self.versions[repo] = None
        
        # pylearn.
        #try:
	#    import pylearn
	#    self.versions['pylearn'] = self._get_hg_version(self._get_module_parent_path(pylearn))
	#except ImportError:
        #    self.versions['pylearn'] = None

        # pylearn2.
        try:
	    import pylearn2
	    self.versions['pylearn2'] = self._get_git_version(self._get_module_parent_path(pylearn2))
	except ImportError:
	    self.versions['pylearn2'] = None

        # Theano.
        try:
	    import theano
	    v = theano.__version__
	    if v == 'unknown':
		v = self._get_git_version(self._get_module_parent_path(theano))
	    self.versions['theano'] = v
	except ImportError:
	    self.versions['theano'] = None

        # Jobman: will only work with old assembla version (there is no version
        # number currently available when running setup.py).
        #try:
	#    import jobman
	#    self.versions['jobman'] = self._get_hg_version(self._get_module_parent_path(jobman))
	#except ImportError:
	#    self.versions['jobman'] = None
	
        # Numpy.
        try:
	    import numpy
	    self.versions['numpy'] = numpy.__version__
	except ImportError:
	    self.versions['numpy'] = None

        # Scipy.
        try:
	    import scipy
	    self.versions['scipy'] = scipy.__version__
	except ImportError:
	    self.versions['scipy'] = None
	    
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

    def print_versions(self):
        """
        Print version of the Python packages as a string.
        e.g. numpy:1.6.1 | pylearn:a6e634b83d | pylearn2:57a156beb0
        """
        print self.__str__()
        
    def print_exp_env_info(self, print_theano_config=False):
	"""
        Return basic information about the experiment setup such as the hostname
        of the machine the experiment was run on, the operating system installed
        on the machine.
        If the switch print_theano_config is set to True, then information about
        the theano configuration will be displayed.
        """
	print 'HOST: ', self.exp_env_info['host']
	print 'CPU: ', self.exp_env_info['cpu']
	print 'OS: ', self.exp_env_info['os']	
	if print_theano_config:
	    print self.exp_env_info['theano_config']

