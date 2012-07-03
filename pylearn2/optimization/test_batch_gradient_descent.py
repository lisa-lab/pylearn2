from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
import theano.tensor as T
from pylearn2.utils import sharedX
import numpy as np
from theano import config
from theano.printing import min_informative_str

def test_batch_gradient_descent():
        """ Verify that batch gradient descent works by checking that
        it minimizes a quadratic function f(x) = x^T A x + b^T x + c
        correctly for several sampled values of A, b, and c.
        The ground truth minimizer is x = np.linalg.solve(A,-b)"""

        n = 3

        A = T.matrix(name = 'A')
        b = T.vector(name = 'b')
        c = T.scalar(name = 'c')

        x = sharedX( np.zeros((n,)) , name = 'x')

        half = np.cast[config.floatX](0.5)

        obj = half * T.dot(T.dot(x,A),x)+T.dot(b,x)+c

        minimizer = BatchGradientDescent(
                        objective = obj,
                        params = [ x],
                        inputs = [ A, b, c])

        num_samples = 3

        rng = np.random.RandomState([1,2,3])

        for i in xrange(num_samples):
            A = np.cast[config.floatX](rng.randn(1.5*n,n))
            A = np.cast[config.floatX](np.dot(A.T,A))
            A += np.cast[config.floatX](np.identity(n) * .02)
            b = np.cast[config.floatX](rng.randn(n))
            c = np.cast[config.floatX](rng.randn())
            x.set_value(np.cast[config.floatX](rng.randn(n)))

            analytical_x = np.linalg.solve(A,-b)

            actual_obj = minimizer.minimize(A,b,c)
            actual_x = x.get_value()

            #Check that the value returned by the minimize method
            #is the objective function value at the parameters
            #chosen by the minimize method
            cur_obj = minimizer.obj(A,b,c)
            assert np.allclose(actual_obj, cur_obj)

            x.set_value(analytical_x)
            analytical_obj = minimizer.obj(A,b,c)

            #make sure the objective function is accurate to first 4 digits
            condition1 = not np.allclose(analytical_obj, actual_obj)
            condition2 = np.abs(analytical_obj-actual_obj) >= 1e-4 * np.abs(analytical_obj)

            if (config.floatX == 'float64' and condition1) \
                    or (config.floatX == 'float32' and condition2):
                print 'objective function value came out wrong on sample ',i
                print 'analytical obj', analytical_obj
                print 'actual obj',actual_obj

                """
                The following section of code was used to verify that numerical
                error can make the objective function look non-convex

                print 'Checking for numerically induced non-convex behavior'
                def f(x):
                    return 0.5 * np.dot(x,np.dot(A,x)) + np.dot(b,x) + c

                x.set_value(actual_x)
                minimizer._compute_grad(A,b,c)
                minimizer._normalize_grad()
                d = minimizer.param_to_grad_shared[x].get_value()

                x = actual_x.copy()
                prev = f(x)
                print prev
                step_size = 1e-4
                x += step_size * d
                cur = f(x)
                print cur
                cur_sgn = np.sign(cur-prev)
                flip_cnt = 0
                for i in xrange(10000):
                    x += step_size * d
                    prev = cur
                    cur = f(x)
                    print cur
                    prev_sgn = cur_sgn
                    cur_sgn = np.sign(cur-prev)
                    if cur_sgn != prev_sgn:
                        print 'flip'
                        flip_cnt += 1
                        if flip_cnt > 1:
                            print "Non-convex!"

                            from matplotlib import pyplot as plt
                            y = []

                            x = actual_x.copy()
                            for j in xrange(10000):
                                y.append(f(x))
                                x += step_size * d

                            plt.plot(y)
                            plt.show()

                            assert False

                print 'None found'
                """

                #print 'actual x',actual_x
                #print 'A:'
                #print A
                #print 'b:'
                #print b
                #print 'c:'
                #print c
                x.set_value(actual_x)
                minimizer._compute_grad(A,b,c)
                x_grad = minimizer.param_to_grad_shared[x]
                actual_grad =  x_grad.get_value()
                correct_grad = 0.5 * np.dot(A,x.get_value())+ 0.5 * np.dot(A.T, x.get_value()) +b
                if not np.allclose(actual_grad, correct_grad):
                    print 'gradient was wrong at convergence point'
                    print 'actual grad: '
                    print actual_grad
                    print 'correct grad: '
                    print correct_grad
                    print 'max difference: ',np.abs(actual_grad-correct_grad).max()
                    assert False


                minimizer._normalize_grad()
                d = minimizer.param_to_grad_shared[x].get_value()
                step_len = ( np.dot(b,d) + 0.5 * np.dot(d,np.dot(A,actual_x)) \
                        + 0.5 * np.dot(actual_x,np.dot(A,d)) ) / np.dot(d, np.dot(A,d))

                g = np.dot(A,actual_x)+b
                deriv = np.dot(g,d)

                print 'directional deriv at actual', deriv
                print 'optimal step_len', step_len
                optimal_x = actual_x - d * step_len
                g = np.dot(A,optimal_x) + b
                deriv = np.dot(g,d)

                print 'directional deriv at optimal: ',deriv
                x.set_value(optimal_x)
                print 'obj at optimal: ',minimizer.obj(A,b,c)



                print 'eigenvalue range:'
                val, vec = np.linalg.eig(A)
                print (val.min(),val.max())
                print 'condition number: ',(val.max()/val.min())
                assert False


if __name__ == '__main__':
    test_batch_gradient_descent()
