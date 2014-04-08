import sys
import warnings
from setuptools import setup, find_packages, Extension
import numpy

if 'develop' not in sys.argv:
    raise NotImplementedError("since Pylearn2 is under rapid, active "
                              "development, `python setup.py install` is "
                              "intentionally disabled to prevent other "
                              "problems. Run `python setup.py develop` to "
                              "install Pylearn2.")
    # Detailed notes:
    # This modification of setup.py is designed to prevent two problems
    # novice users frequently encountered:
    # 1) Novice users frequently used "git clone" to get a copy of Pylearn2,
    # then ran setup.py install, then would use "git pull" to get a bug fix
    # but would forget to run "setup.py install" again.
    # 2) Novice users frequently used "sudo" to make an "installed" copy of
    # Pylearn2, then try to use the tutorials in the "scripts" directory in
    # the "installed" copy. Since the tutorials are then in a directory owned
    # by root and need to create files in the local directory, some users
    # would run the tutorials using "sudo". Besides being dangerous, this
    # created additional problems because "sudo" does not just run the script
    # with root privileges, it actually changes the user to root, and thus
    # pylearn2-related environment variables configured in the user's
    # .bashrc would no longer be available.
    # Installing only in development mode avoids both problems because there
    # is now only a single copy of the code and it is stored in a directory
    # editable by the user.
    # Note that none of the Pylearn2 installation documentation recommends
    # using setup.py install or pip. Most of the Pylearn2 developers just
    # obtain Pylearn2 via git clone and then add it to their PYTHONPATH
    # manually.

# Because many people neglected to run the pylearn2/utils/setup.py script
# separately, we compile the necessary Cython extensions here but because
# Cython is not a strict dependency, we issue a warning when it is not
# available.
try:
    from Cython.Distutils import build_ext
    cython_available = True
except ImportError:
    warnings.warn("Cython was not found and hence pylearn2.utils._window_flip "
                  "and pylearn2.utils._video and classes that depend on them "
                  "(e.g. pylearn2.train_extensions.window_flip) will not be "
                  "available")
    cython_available = False

if cython_available:
    cmdclass = {'build_ext': build_ext}
    ext_modules = [Extension("pylearn2.utils._window_flip",
                             ["pylearn2/utils/_window_flip.pyx"],
                             include_dirs=[numpy.get_include()]),
                   Extension("pylearn2.utils._video",
                             ["pylearn2/utils/_video.pyx"],
                             include_dirs=[numpy.get_include()])]
else:
    cmdclass = {}
    ext_modules = []

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    name='pylearn2',
    version='0.1dev',
    packages=find_packages(),
    description='A machine learning library built on top of Theano.',
    license='BSD 3-clause license',
    long_description=open('README.rst').read(),
    install_requires=['numpy>=1.5', 'theano', 'pyyaml', 'argparse'],
    package_data={
        '': ['*.cu', '*.cuh', '*.h'],
    },
)
