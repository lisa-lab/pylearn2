"""
A theano / pylearn2 wrapper for Alex Krizhevsky's cuda-convnet
software.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

"""
This module contains code copied directly or modified from cuda-convnet.
The copyright and licensing notice for this code is reproduced below:


/*
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

"""

import errno
import logging
import os
import shutil
import stat
import sys

from theano import config
from theano.gof.cmodule import get_lib_extension
from theano.gof.compilelock import get_lock, release_lock
from theano.sandbox import cuda
from theano.sandbox.cuda import nvcc_compiler

from shared_code import this_dir

_logger_name = 'pylearn2.sandbox.cuda_convnet'
_logger = logging.getLogger(_logger_name)
#_logger.addHandler(logging.StreamHandler())
#_logger.setLevel(logging.DEBUG)

_logger.debug('importing')


if cuda.cuda_available:

    cuda_convnet_loc = os.path.join(config.compiledir, 'cuda_convnet')
    # In partial dependency order: the last ones depend on the first ones
    cuda_convnet_file_roots = ('nvmatrix_kernels', 'nvmatrix', 'conv_util',
                               'filter_acts', 'img_acts', 'weight_acts')
    cuda_convnet_so = os.path.join(cuda_convnet_loc,
            'cuda_convnet.' + get_lib_extension())
    libcuda_convnet_so = os.path.join(cuda_convnet_loc,
            'libcuda_convnet.' + get_lib_extension())

    def should_recompile():
        """
        Returns True if the .so files are not present or outdated.
        """
        # The following list is in alphabetical order.
        source_files = (
                'conv_util.cu',
                'conv_util.cuh',
                'cudaconv2.cuh',
                'filter_acts.cu',
                'img_acts.cu',
                'nvmatrix.cu',
                'nvmatrix.cuh',
                'nvmatrix_kernels.cu',
                'nvmatrix_kernels.cuh',
                'nvmatrix_operators.cuh',
                'weight_acts.cu')
        stat_times = [
                os.stat(os.path.join(this_dir, source_file))[stat.ST_MTIME]
                for source_file in source_files]
        date = max(stat_times)
        _logger.debug('max date: %f', date)

        if (not os.path.exists(cuda_convnet_so) or
                date >= os.stat(cuda_convnet_so)[stat.ST_MTIME]):
            return True

        return False

    # Compile .cu files in cuda_convnet
    _logger.debug('nvcc_compiler.rpath_defaults: %s',
            str(nvcc_compiler.rpath_defaults))
    import time
    t1 = time.time()
    if should_recompile():
        _logger.debug('should recompile')

        # Concatenate all .cu files into one big mod.cu
        code = []
        for file_root in cuda_convnet_file_roots:
            source_file = file_root + '.cu'
            code.append(open(os.path.join(this_dir, source_file)).read())
        code = '\n'.join(code)

        get_lock()
        try:
            # Check if the compilation has already been done by another process
            # while we were waiting for the lock
            if should_recompile():
                _logger.debug('recompiling')

                try:
                    compiler = nvcc_compiler.NVCC_compiler()
                    args = compiler.compile_args()

                    # compiler.compile_args() can execute a
                    # compilation This currently will remove empty
                    # directory in the compile dir.  So we must make
                    # destination directory after calling it.
                    if not os.path.exists(cuda_convnet_loc):
                        os.makedirs(cuda_convnet_loc)
                    compiler.compile_str('cuda_convnet',
                            code,
                            location=cuda_convnet_loc,
                            include_dirs=[this_dir],
                            lib_dirs=nvcc_compiler.rpath_defaults,  # ???
                            libs=['cublas'],
                            preargs=['-O3'] + args,
                            py_module=False)
                except Exception, e:
                    _logger.error("Failed to compile %s.cu: %s",
                                  file_root, str(e))
            else:
                _logger.debug('already compiled by another process')

        finally:
            release_lock()
    else:
        _logger.debug('not recompiling')

    # If necessary, create a symlink called libcuda_convnet.so
    def ok():
        """
        Check if an existing library exists and can be read.
        """
        try:
            open(libcuda_convnet_so).close()
            return True
        except IOError:
            return False

    if not ok():
        if sys.platform == "win32":
            # The Python `os` module does not support symlinks on win32.
            shutil.copyfile(cuda_convnet_so, libcuda_convnet_so)
        else:
            try:
                os.symlink(cuda_convnet_so, libcuda_convnet_so)
            except OSError, e:
                # This may happen for instance when running multiple
                # concurrent jobs, if two of them try to create the
                # symlink simultaneously.
                # If that happens, we verify that the existing symlink is
                # indeed working.
                if getattr(e, 'errno', None) != errno.EEXIST or not ok():
                    raise

    # Raise an error if libcuda_convnet_so is still not available
    open(libcuda_convnet_so).close()

    # Add cuda_convnet to the list of places that are hard-coded into
    # compiled modules' runtime library search list.
    nvcc_compiler.add_standard_rpath(cuda_convnet_loc)

    t2 = time.time()
    _logger.debug('successfully imported. Compiled in %fs', t2 - t1)
