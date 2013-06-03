import copy
import theano
import numpy as np
import theano.tensor as T
from theano import config
from theano import function
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b
from pylearn2.sandbox.cuda_convnet.probabilistic_max_pooling import  prob_max_pool_c01b
from pylearn2.utils import float32_floatX
from pylearn2.testing.skip import skip_if_no_gpu

skip_if_no_gpu()

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

@float32_floatX
def test_correctness():
    """
    Test the forward pass Op against theano graph implementation
    """

    rng = np.random.RandomState([2012,7,19])
    batch_size_list = [1, 5]
    channels = 16
    rows_list = [2, 24]
    pool_rows_list = [2, 3]

    # TODO theano graph version fails with pool shape 1,1,
    # try it with python version

    for batch_size in batch_size_list:
        for rows, pool_rows in zip(rows_list, pool_rows_list):
            cols = rows
            pool_cols = pool_rows

            zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)

            z = T.tensor4()

            # gpu op
            p, h = prob_max_pool_c01b(z, (pool_rows, pool_cols) )
            func = function([z], [p, h], mode = mode_with_gpu)

            p_op, h_op = func(zv)

            # theano graph
            p, h = max_pool_c01b(z, (pool_rows, pool_cols) )
            func = function([z], [p, h], mode = mode_without_gpu)

            p_th, h_th = func(zv)

            assert np.allclose(p_op, p_th)
            assert np.allclose(h_op, h_th)

@float32_floatX
def test_top_donw_correctness():
    """
    Test the forward pass Op against theano graph implementation
    """

    rng = np.random.RandomState([2012,7,19])
    batch_size_list = [1]
    channels = 16
    rows_list = [2, 24]
    pool_rows_list = [2, 3]

    # TODO theano graph version fails with pool shape 1,1,
    # try it with python version

    for batch_size in batch_size_list:
        for rows, pool_rows in zip(rows_list, pool_rows_list):
            cols = rows
            pool_cols = pool_rows

            zv = rng.randn(channels, rows, cols, batch_size).astype(config.floatX)
            tv = rng.randn(channels, rows / pool_rows, cols / pool_cols, batch_size).astype(config.floatX)

            z = T.tensor4()
            t = T.tensor4()

            # gpu op
            p, h = prob_max_pool_c01b(z, (pool_rows, pool_cols), top_down = t)
            func = function([z, t], [p, h], mode = mode_with_gpu)

            p_op, h_op = func(zv, tv)

            # theano graph
            p, h = max_pool_c01b(z, (pool_rows, pool_cols), top_down = t)
            func = function([z, t], [p, h], mode = mode_without_gpu)

            p_th, h_th = func(zv, tv)

            assert np.allclose(p_op, p_th)
            assert np.allclose(h_op, h_th)

@float32_floatX
def test_grad():
    """
    Test Op's gradient w.r.t top_down against theano graph implementation
    """

    rng = np.random.RandomState([2012,7,19])
    batch_size_list = [1]
    channels = 16
    rows_list = [2, 24]
    pool_rows_list = [2, 3]

    # TODO theano graph version fails with pool shape 1,1,
    # try it with python version

    for batch_size in batch_size_list:
        for rows, pool_rows in zip(rows_list, pool_rows_list):
            cols = rows
            pool_cols = pool_rows

            zv = rng.randn(channels, rows, cols,
                    batch_size).astype(config.floatX)
            tv = rng.randn(channels, rows / pool_rows, cols / \
                    pool_cols, batch_size).astype(config.floatX)

            z = T.tensor4()
            t = T.tensor4()

            # gpu op
            p, h = prob_max_pool_c01b(z, (pool_rows, pool_cols), top_down = t)
            gh_t = T.grad(h.sum(), t)
            gp_t = T.grad(p.sum(), t)
            gh_z = T.grad(h.sum(), z)
            gp_z = T.grad(p.sum(), z)
            gph_z = T.grad(p.sum() + h.sum(), z)
            gph_t = T.grad(p.sum() + h.sum(), t)

            func = function([z, t], [gh_t, gp_t, gh_z, gp_z, gph_z, gph_t],
                                mode = mode_with_gpu)
            op_rval = func(zv, tv)

            # theano graph
            p, h = max_pool_c01b(z, (pool_rows, pool_cols) , top_down = t)
            gh_t = T.grad(h.sum(), t)
            gp_t = T.grad(p.sum(), t)
            gh_z = T.grad(h.sum(), z)
            gp_z = T.grad(p.sum(), z)
            gph_z = T.grad(p.sum() + h.sum(), z)
            gph_t = T.grad(p.sum() + h.sum(), t)

            func = function([z, t], [gh_t, gp_t, gh_z, gp_z, gph_z, gph_t],
                                mode = mode_without_gpu)
            th_rval = func(zv, tv)

            for op, th in zip (op_rval, th_rval):
                assert np.allclose(op, th, rtol=1e-04, atol=1e-06)

