from pylearn2.models.model import Model
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import FixedVarDescr
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
import theano.tensor as T

class DeconvNet(Model):
    """
    A deconvolutional network.
    TODO add paper reference
    """

    def __init__(self, batch_size, input_shape, input_channels,
            hid_shape, hid_channels):

        self.__dict__.update(locals())
        del self.self

        self.input_space = Conv2DSpace(input_shape, input_channels)
        self.output_space = Conv2DSpace(hid_shape, hid_channels)

    def censor_updates(self, updates):
        """
        Modify the updates proposed by the training algorithm to
        preserve the norm constraint on the kernels.
        """

        if self.W in updates:
            update = updates[self.W]
            norms = T.sqrt(T.sqr(updates).sum(axis=(1,2,3)))
            update = update / (1e-7 + norms)
            update[self.W] = update

class InferenceCallback(object):

    def __init__(self, model, code):
        self.__dict__.update(locals())
        del self.self

    def __call__(self, X, Y):
        """
        updates self.code

        X: a numpy tensor for the input image of shape (batch_size, rows, cols, channels)
        Y: unused

        the model is available as self.model
        """


        raise NotImplementedError()

class DeconvNetMSESparsity(Cost):
    """
    The standard cost for training a deconvolution network.

    """

    def __call__(self, model, X, Y=None, code=None, **kwargs):
        """
            Returns a theano expression for the mean squared error.

            model: a DeconvNet instance
            X: a theano tensor of shape (batch_size, rows, cols, channels)
            Y: unused
            deconv_net_code: the theano shared variable representing the deconv net's code
            kwargs: unused
        """

        # Training algorithm should always supply the code
        assert code is not None

        # We need to return the mean squared error
        raise NotImplementedError()

    def get_fixed_var_descr(self, model, X, Y):
        """
            Returns a FixedVarDescr describing how to update the code.
            We use this mechanism because it is the easiest way to use a python
            loop to do inference.

            model: a DeconvNet instance
            X: a theano tensor of shape (batch_size, rows, cols, channels)
            Y: unused
        """

        rval = FixedVarDescr()

        code = sharedX(model.output_space.get_origin_batch(model.batch_size))

        rval.fixed_vars = {'deconv_net_code' : code}

        rval.on_load_batch = [InferenceCallback(model, code)]

        return rval
