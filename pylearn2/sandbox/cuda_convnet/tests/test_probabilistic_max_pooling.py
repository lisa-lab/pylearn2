import copy
import theano
import numpy as np
import theano.tensor as T
from theano import config
from theano import function
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b
from pylearn2.sandbox.cuda_convnet.probabilistic_max_pooling import max_pool_c01b as max_pool_op

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
            'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

#The CPU tests already compare C/Py, so we only check C/GPU
mode_with_gpu = copy.copy(mode_with_gpu)
mode_without_gpu = copy.copy(mode_without_gpu)
mode_with_gpu.check_py_code = False
mode_without_gpu.check_py_code = False


def test_correctness():
    """
    Test the forward pass Op against theano graph implementation
    """

    rng = np.random.RandomState([2012,7,19])
    batch_size_list = [1, 5, 128]
    channels = 16
    rows_list = [2, 8, 30]
    pool_rows_list = [2, 4, 3]

    # TODO theano graph version fails with pool shape 1,1,
    # try it with python version

    for batch_size in batch_size_list:
        for rows, pool_rows in zip(rows_list, pool_rows_list):
            cols = rows
            pool_cols = pool_rows

            zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)

            z = T.tensor4()

            # gpu op
            p, h = max_pool_op(z, (pool_rows, pool_cols) )
            func = function([z], [p, h], mode = mode_with_gpu)

            p_op, h_op = func(zv)

            # theano graph
            p, h = max_pool_c01b(z, (pool_rows, pool_cols) )
            func = function([z], [p, h], mode = mode_without_gpu)

            p_th, h_th = func(zv)

            assert np.allclose(p_op, p_th)
            assert np.allclose(h_op, h_th)

def test_grad_correctness():
    """
    Test Op's gradient against theano graph implementation
    """

    rng = np.random.RandomState([2012,7,19])
    batch_size_list = [1, 5, 128]
    channels = 16
    rows_list = [2, 8, 30]
    pool_rows_list = [2, 4, 3]

    #TODO theano graph version fails with pool shape 1,1,
    # try it with python version

    for batch_size in batch_size_list:
        for rows, pool_rows in zip(rows_list, pool_rows_list):
            cols = rows
            pool_cols = pool_rows

            zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)

            z = T.tensor4()

            # gpu op
            p, h = max_pool_op(z, (pool_rows, pool_cols) )
            gz = T.grad(h.sum() + p.sum(), z)
            func = function([z], gz, mode = mode_with_gpu)

            op_gz = func(zv)

            # theano graph
            p, h = max_pool_c01b(z, (pool_rows, pool_cols) )
            gz = T.grad(h.sum() + p.sum(), z)
            func = function([z], gz, mode = mode_without_gpu)

            th_gz = func(zv)

            assert np.allclose(op_gz, th_gz, rtol=1e-04, atol=1e-06)

def test_top_down_grad_correctness():
    """
    Test Op's gradient w.r.t top_down against theano graph implementation
    """

    rng = np.random.RandomState([2012,7,19])
    batch_size_list = [128]
    channels = 16
    rows_list = [2, 8, 20]
    pool_rows_list = [2, 4, 5]

    # TODO theano graph version fails with pool shape 1,1,
    # try it with python version

    # TODO the results doesn't match for (30,30), (3, 3)
    # check verify grad

    for batch_size in batch_size_list:
        for rows, pool_rows in zip(rows_list, pool_rows_list):
            cols = rows
            pool_cols = pool_rows

            zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)
            tv = rng.randn(channels, rows / pool_rows, cols / pool_cols, batch_size).astype(config.floatX)

            z = T.tensor4()
            t = T.tensor4()

            # gpu op
            p, h = max_pool_op(z, (pool_rows, pool_cols), top_down = t)
            gt = T.grad(h.sum() + p.sum(), t)
            gt = T.grad(h.sum() + p.sum(), t)
            func = function([z, t], gt, mode = mode_with_gpu)

            op_gt = func(zv, tv)

            # theano graph
            p, h = max_pool_c01b(z, (pool_rows, pool_cols) , top_down = t)
            gt = T.grad(h.sum() + p.sum(), t)
            func = function([z, t], gt, mode = mode_without_gpu)

            th_gt = func(zv, tv)

            print batch_size, rows, pool_rows
            assert np.allclose(op_gt, th_gt, rtol=1e-04, atol=1e-06)

if __name__ == "__main__":
    #test_correctness()
    #test_grad_correctness()
    test_top_down_grad_correctness()
