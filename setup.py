import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
import numpy

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


# Inform user of setup.py develop preference
class pylearn2_install(install):
    def run(self):
        print ("Because Pylearn2 is under heavy development, we generally do "
               "not advice using the `setup.py install` command. Please "
               "consider using the `setup.py develop` command instead for the "
               "following reasons:\n\n1. Using `setup.py install` creates a "
               "copy of the Pylearn2 source code in your Python installation "
               "path. In order to update Pylearn2 afterwards you will need to "
               "rerun `setup.py install` (!). Simply using `git pull` to "
               "update your local copy of Pylearn2 code will not suffice. \n\n"
               "2. When using `sudo` to install Pylearn2, all files, "
               "including the tutorials, will be copied to a directory owned "
               "by root. Not only is running tutorials as root unsafe, it "
               "also means that all Pylearn2-related environment variables "
               "which were defined for the user will be unavailable.\n\n"
               "Pressing enter will continue the installation of Pylearn2 in "
               "`develop` mode instead. Note that this means that you need to "
               "keep this folder with the Pylearn2 code in its current "
               "location. If you know what you are doing, and are very sure "
               "that you want to install Pylearn2 using the `install` "
               "command instead, please type `install`.\n")
        mode = None
        while mode not in ['', 'install', 'develop', 'cancel']:
            if mode is not None:
                print("Please try again")
            mode = raw_input("Installation mode: [develop]/install/cancel: ")
        if mode in ['', 'develop']:
            self.distribution.run_command('develop')
        if mode == 'install':
            return install.run(self)
cmdclass.update({'install': pylearn2_install})

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    name='pylearn2',
    version='0.1dev',
    packages=find_packages(),
    description='A machine learning library built on top of Theano.',
    license='BSD 3-clause license',
    long_description=open('README.rst').read(),
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    install_requires=['numpy>=1.5', 'pyyaml', 'argparse', "Theano"],
    package_data={
        '': ['*.cu', '*.cuh', '*.h'],
    },
)
