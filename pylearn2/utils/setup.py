from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("_window_flip", ["_window_flip.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension("_video", ["_video.pyx"],
                  include_dirs=[numpy.get_include()])
    ]
)
