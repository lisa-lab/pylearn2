from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("pylearn2.utils._window_flip",
                  ["pylearn2/utils/_window_flip.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("pylearn2.utils._video",
                  ["pylearn2/utils/_video.pyx"],
                  include_dirs=[numpy.get_include()])
    ]
)

from setuptools import setup, find_packages

setup(
    name='pylearn2',
    version='0.1dev',
    packages=find_packages(),
    description='A machine learning library build on top of Theano.',
    license='BSD 3-clause license',
    long_description=open('README.rst').read(),
    install_requires=['numpy>=1.5', 'theano', 'pyyaml', 'argparse'],
    package_data={
        '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.h', '*.so'],
    },
)
