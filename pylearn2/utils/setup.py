from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np
include_dirs = [np.get_include()]

setup(
  name='pylearn2 utilities',
  cmdclass={'build_ext': build_ext},
  ext_modules=[
    Extension("_video",
              ["_video.pyx"],
              include_dirs=include_dirs),
  ]
)
