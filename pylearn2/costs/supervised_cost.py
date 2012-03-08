import theano.tensor as T
import warnings
from pylearn2.costs.error import SupervisedError

class CrossEntropy(SupervisedError):
    def __call__(self, model, X, Y):
        return (
            - Y * T.log(model(X)) -
            (1 - Y) * T.log(model(X))
            ).sum(axis=1).mean()

#Elemwise{Composite{[Composite{[true_div(neg(i0), mul(i1, i2))]}(sub(i0, i1), i2, i3)]}}(TensorConstant{(1, 1) of 1.0}, UnsupervisedExhaustiveSGD[Y], <TensorType(float64, (True, True))>, <TensorType(float64, matrix)>), [Elemwise{Composite{[Composite{[true_div(neg(i0), mul(i1, i2))]}(sub(i0, i1), i2, i3)]}}.0])