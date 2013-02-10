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

_logger_name = 'pylearn2.sandbox.cuda_convnet.convnet_compile'
_logger = logging.getLogger(_logger_name)
#_logger.addHandler(logging.StreamHandler())
#_logger.setLevel(logging.DEBUG)

_logger.debug('importing')


cuda_convnet_loc = os.path.join(config.compiledir, 'cuda_convnet')
# In partial dependency order: the last ones depend on the first ones
cuda_convnet_file_roots = ('nvmatrix_kernels', 'nvmatrix', 'conv_util',
                           'filter_acts', 'img_acts', 'weight_acts')
cuda_convnet_so = os.path.join(cuda_convnet_loc,
        'cuda_convnet.' + get_lib_extension())
libcuda_convnet_so = os.path.join(cuda_convnet_loc,
        'libcuda_convnet.' + get_lib_extension())


def convnet_available():
    # If already compiled, OK
    if convnet_available.compiled:
        _logger.debug('already compiled')
        return True

    # If there was an error, do not try again
    if convnet_available.compile_error:
        _logger.debug('error last time')
        return False

    # Else, we need CUDA
    if not cuda.cuda_available:
        convnet_available.compile_error = True
        _logger.debug('cuda unavailable')
        return False

    # Try to actually compile
    success = convnet_compile()
    if success:
        convnet_available.compiled = True
    else:
        convnet_available.compile_error = False
    _logger.debug('compilation success: %s', success)

    return convnet_available.compiled

# Initialize variables in convnet_available
convnet_available.compiled = False
convnet_available.compile_error = False


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


def symlink_ok():
    """
    Check if an existing library exists and can be read.
    """
    try:
        open(libcuda_convnet_so).close()
        return True
    except IOError:
        return False


def convnet_compile():
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
                    return False
            else:
                _logger.debug('already compiled by another process')

        finally:
            release_lock()
    else:
        _logger.debug('not recompiling')

    # If necessary, create a symlink called libcuda_convnet.so
    if not symlink_ok():
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
                if (getattr(e, 'errno', None) != errno.EEXIST
                        or not symlink_ok()):
                    raise

    # Raise an error if libcuda_convnet_so is still not available
    open(libcuda_convnet_so).close()

    # Add cuda_convnet to the list of places that are hard-coded into
    # compiled modules' runtime library search list.
    nvcc_compiler.add_standard_rpath(cuda_convnet_loc)

    t2 = time.time()
    _logger.debug('successfully imported. Compiled in %fs', t2 - t1)

    return True
