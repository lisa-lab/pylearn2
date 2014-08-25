import numpy
import theano
import theano.tensor as T
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Linear, ConvRectifiedLinear
from pylearn2.models.vae import VAE
from pylearn2.models.vae.visible import BinaryVisible
from pylearn2.models.vae.latent import DiagonalGaussianPrior
from pylearn2.space import Conv2DSpace


def test_one_sample_allowed():
    """
    VAE allows one sample per data point
    """
    encoding_model = MLP(nvis=10, layers=[Linear(layer_name='h', dim=10,
                                                 irange=0.01)])
    decoding_model = MLP(nvis=5, layers=[Linear(layer_name='h', dim=10,
                                                irange=0.01)])
    visible = BinaryVisible(decoding_model=decoding_model)
    latent = DiagonalGaussianPrior(encoding_model=encoding_model,
                                   num_samples=1)
    vae = VAE(nvis=10, visible=visible, latent=latent, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X)
    f = theano.function(inputs=[X], outputs=lower_bound)
    f(numpy.random.uniform(size=(10, 10)))


def test_multiple_samples_allowed():
    """
    VAE allows multiple samples per data point
    """
    encoding_model = MLP(nvis=10, layers=[Linear(layer_name='h', dim=10,
                                                 irange=0.01)])
    decoding_model = MLP(nvis=5, layers=[Linear(layer_name='h', dim=10,
                                                irange=0.01)])
    visible = BinaryVisible(decoding_model=decoding_model)
    latent = DiagonalGaussianPrior(encoding_model=encoding_model,
                                   num_samples=10)
    vae = VAE(nvis=10, visible=visible, latent=latent, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X)
    f = theano.function(inputs=[X], outputs=lower_bound)
    f(numpy.random.uniform(size=(10, 10)))


def test_convolutional_compatible():
    """
    VAE allows convolutional encoding networks
    """
    encoding_model = MLP(
        input_space=Conv2DSpace(shape=[4, 4], num_channels=1),
        layers=[ConvRectifiedLinear(
            layer_name='h',
            output_channels=2,
            kernel_shape=[2, 2],
            kernel_stride=[1, 1],
            pool_shape=[1, 1],
            pool_stride=[1, 1],
            pool_type='max',
            irange=0.01
        )]
    )
    decoding_model = MLP(nvis=5, layers=[Linear(layer_name='h', dim=16,
                                                irange=0.01)])
    visible = BinaryVisible(decoding_model=decoding_model)
    latent = DiagonalGaussianPrior(encoding_model=encoding_model,
                                   num_samples=10)
    vae = VAE(nvis=16, visible=visible, latent=latent, nhid=5)
    X = T.matrix('X')
    lower_bound = vae.log_likelihood_lower_bound(X)
    f = theano.function(inputs=[X], outputs=lower_bound)
    f(numpy.random.uniform(size=(10, 16)))
