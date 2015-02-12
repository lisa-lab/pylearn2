"""
Factored Gated AutoEncoder Class.

This code is adapted from the implementation of Roland Memisevic,
specifically from the paper: "Gradient-based learning of higher-order
image features" (http://www.iro.umontreal.ca/~memisevr/code/rae/index.html).
And the autoencoder implementation of Pylearn2.
"""

__authors__ = "Raul Peralta Lozada"
__copyright__ = "(c) 2014, University of Edinburgh"
__license__ = "3-clause BSD License"
__contact__ = "raulpl25@gmail.com"

# Third-party imports
import numpy
from theano import tensor

# Local imports
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng, make_theano_rng
from pylearn2.models import Model
from pylearn2.blocks import Block
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils import is_iterable
from pylearn2.utils import wraps


class GatedAutoencoder(Block, Model):
    """
    Base class for Gated Autoencoder. You should not use this class
    directly, use one of its subclasses.

    Parameters
    ----------
    nvisX : int
        Number of visible units in the first image.
    nvisY : int
        Number of visible units in the second image.
    nmap : int
        Number of mappings units.
    recepF: tuple
        Size of the receptive field of the images.
    color : bool
        If the images are grayscale or color
    act_enc : callable or string
        Activation function (elementwise nonlinearity) to use for the
        encoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
        functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
        for linear units.
    act_dec : callable or string
        Activation function (elementwise nonlinearity) to use for the
        decoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
        functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
        for linear units.
    irange : float, optional
        Width of the initial range around 0 from which to sample initial
        values for the weights.
    rng : RandomState object or seed, optional
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """
    def __init__(self, nvisX, nvisY, nmap, recepF, act_enc,
                 act_dec, color=True, irange=1e-3, rng=9001):
        # super(GatedAutoencoder, self).__init__()
        Block.__init__(self)
        Model.__init__(self)
        assert nvisX > 0, "Number of visible units must be non-negative"
        assert nvisY > 0, "Number of visible units must be non-negative"
        assert nmap > 0, "Number of mapping units must be positive"
        assert is_iterable(recepF), "recepF must be iterable"
        assert len(recepF) == 2, "Size of the window must be 2-dim"

        self.input_space = VectorSpace((nvisX + nvisY))
        self.output_space = VectorSpace(nmap)
        if color:
            self.channels = 3
        else:
            self.channels = 1

        # Parameters
        self.nvisX = nvisX
        self.nvisY = nvisY
        self.nmap = nmap
        self.irange = irange
        self.rng = make_np_rng(rng, which_method="randn")
        self.recepF = recepF
        self._initialize_visbiasX(nvisX)  # self.visbiasX
        self._initialize_visbiasY(nvisY)  # self.visbiasY
        self._initialize_mapbias()  # self.mapbias

        seed = int(self.rng.randint(2 ** 30))
        self.s_rng = make_theano_rng(seed, which_method="uniform")

        def _resolve_callable(conf, conf_attr):
            if conf[conf_attr] is None or conf[conf_attr] == "linear":
                return None
            # If it's a callable, use it directly.
            if hasattr(conf[conf_attr], '__call__'):
                return conf[conf_attr]
            elif (conf[conf_attr] in globals()
                  and hasattr(globals()[conf[conf_attr]], '__call__')):
                return globals()[conf[conf_attr]]
            elif hasattr(tensor.nnet, conf[conf_attr]):
                return getattr(tensor.nnet, conf[conf_attr])
            elif hasattr(tensor, conf[conf_attr]):
                return getattr(tensor, conf[conf_attr])
            else:
                raise ValueError("Couldn't interpret %s value: '%s'" %
                                (conf_attr, conf[conf_attr]))

        self.act_enc = _resolve_callable(locals(), 'act_enc')
        self.act_dec = _resolve_callable(locals(), 'act_dec')

    def _initialize_mapbias(self):
        """
        Initialize the biases of the mapping units.
        """
        self.mapbias = sharedX(
            numpy.zeros(self.nmap),
            name='mb',
            borrow=True
        )

    def _initialize_visbiasX(self, nvisX):
        """
        Initialize the biases of the first set of visible units.
        """
        self.visbiasX = sharedX(
            numpy.zeros(nvisX),
            name='vbX',
            borrow=True
        )

    def _initialize_visbiasY(self, nvisY):
        """
        Initialize the biases of the second set of visible units.
        """
        self.visbiasY = sharedX(
            numpy.zeros(nvisY),
            name='vbY',
            borrow=True
        )

    get_input_space = Model.get_input_space
    get_output_space = Model.get_output_space


class FactoredGatedAutoencoder(GatedAutoencoder):
    """
    Factored Gated Autoencoder model.

    Parameters
    ----------
    nvisX : int
        Number of visible units in the first image.
    nvisY : int
        Number of visible units in the second image.
    nfac : int
        Number of factors used to project the images.
    nmap : int
        Number of mappings units.
    recepF: tuple
        Size of the receptive field of the images.
    color : bool
        If the images are grayscale or color
    act_enc : callable or string
        Activation function (elementwise nonlinearity) to use for the
        encoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
        functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
        for linear units.
    act_dec : callable or string
        Activation function (elementwise nonlinearity) to use for the
        decoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
        functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
        for linear units.
    irange : float, optional
        Width of the initial range around 0 from which to sample initial
        values for the weights.
    rng : RandomState object or seed, optional
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """

    def __init__(self, nvisX, nvisY, nfac, nmap, recepF, act_enc,
                 act_dec, color=True, irange=1e-3, rng=9001):
        super(FactoredGatedAutoencoder, self).__init__(
            nvisX,
            nvisY,
            nmap,
            recepF,
            act_enc,
            act_dec,
            color,
            irange,
            rng,
        )

        # Parameters
        self.nfac = nfac
        self._initialize_wxf(nvisX, nfac)  # self.wxf
        self._initialize_wyf(nvisY, nfac)  # self.wyf
        self._initialize_whf(nmap, nfac)  # self.whf
        self._initialize_whf_in(nmap, nfac)  # self.whf_in
        self.view_converter = DefaultViewConverter(
            shape=(self.recepF[0], self.recepF[1], self.channels))

        self._params = [self.wxf, self.wyf, self.whf_in, self.whf,
                        self.mapbias, self.visbiasX, self.visbiasY]

    def _initialize_wxf(self, nvisX, nfac, rng=None, irange=None):
        """
        Creation of weight matrix wxf.
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.wxf = sharedX(
            (rng.randn(nvisX, nfac)) * irange,
            name='wxf',
            borrow=True
        )

    def _initialize_wyf(self, nvisY, nfac, rng=None, irange=None):
        """
        Creation of weight matrix wyf.
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.wyf = sharedX(
            (rng.randn(nvisY, nfac)) * irange,
            name='wyf',
            borrow=True
        )

    def _initialize_whf(self, nmap, nfac, rng=None, irange=None):
        """
        Creation of encoding weight matrix whf.
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.whf = sharedX(
            (rng.randn(nmap, nfac)) * irange,
            name='whf',
            borrow=True
        )

    def _initialize_whf_in(self, nmap, nfac, rng=None, irange=None):
        """
        Creation of decoding weight matrix whf.
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.whf_in = sharedX(
            (rng.randn(nmap, nfac)) * irange,
            name='whf_in',
            borrow=True
        )

    def _factorsX(self, inputs):
        return tensor.dot(inputs[:, :self.nvisX], self.wxf)

    def _factorsY(self, inputs):
        return tensor.dot(inputs[:, self.nvisY:], self.wyf)

    def _mappings(self, inputs):
        return self.mapbias + tensor.dot(
            self._factorsX(inputs) * self._factorsY(inputs), self.whf_in.T)

    def _hidden_activation(self, inputs):
        """
        Single minibatch activation function.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing the input minibatch
            that consists of the two images (nviX, nvisY).

        Returns
        -------
        y : tensor_like
            (Symbolic) hidden unit activations given the input.
        """
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._mappings(inputs))

    def encode(self, inputs):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        encoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after encoding (mapping units).
        """
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def _factorsH(self, inputs):
        return tensor.dot(self.encode(inputs), self.whf)

    def decodeX(self, inputs):
        return self.visbiasX + tensor.dot(
            self._factorsY(inputs) * self._factorsH(inputs), self.wxf.T)

    def decodeY(self, inputs):
        return self.visbiasY + tensor.dot(
            self._factorsX(inputs) * self._factorsH(inputs), self.wyf.T)

    def reconstructX(self, inputs):
        """
        Reconstruction of input x.

        Parameters
        ----------
        inputs : tensor_like
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded
        """
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        return act_dec(self.decodeX(inputs))

    def reconstructY(self, inputs):
        """
        Reconstruction of input y.

        Parameters
        ----------
        inputs : tensor_like
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded
        """
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        return act_dec(self.decodeY(inputs))

    def reconstructXY(self, inputs):
        """
        Reconstruction of both images.

        Parameters
        ----------
        inputs : tensor_like
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).
        """
        return (self.reconstructX(inputs),
                self.reconstructY(inputs))

    def __call__(self, inputs):
        """
        This just aliases the `encode()` function for syntactic
        sugar/convenience.
        """
        return self.encode(inputs)

    @wraps(GatedAutoencoder._modify_updates)
    def _modify_updates(self, updates):
        """
        The filters have to be normalized in each update to
        increase their stability.
        """
        wxf = self.wxf
        wyf = self.wyf
        wxf_updated = updates[wxf]
        wyf_updated = updates[wyf]
        nwxf = (wxf_updated.std(0) + 0.000001)[numpy.newaxis, :]
        nwyf = (wyf_updated.std(0) + 0.000001)[numpy.newaxis, :]
        meannxf = nwxf.mean()
        meannyf = nwyf.mean()
        # Center filters
        centered_wxf = wxf_updated - wxf_updated.mean(0)
        centered_wyf = wyf_updated - wyf_updated.mean(0)
        # Fix standard deviation
        wxf_updated = centered_wxf*(meannxf/nwxf)
        wyf_updated = centered_wyf*(meannyf/nwyf)
        updates[wxf] = wxf_updated
        updates[wyf] = wyf_updated

    def get_weights_topo(self):
        """
        Returns a topological view of the weights.

        Returns
        -------
        weights : ndarray
            Same as the return value of `get_weights` but formatted as a 4D
            tensor with the axes being (hidden/factor units, rows, columns,
            channels).The the number of channels is either 1 or 3
            (because they will be visualized as grayscale or RGB color).
            At the moment the function only supports factors whose sqrt
            is exact.
        """
        wxf = self.wxf.get_value(borrow=False).T
        wyf = self.wyf.get_value(borrow=False).T
        wxf_view = self.view_converter.design_mat_to_weights_view(wxf)
        wyf_view = self.view_converter.design_mat_to_weights_view(wyf)
        h = int(numpy.ceil(numpy.sqrt(self.nfac)))
        new_weights = numpy.zeros((
                                  wxf_view.shape[0]*2,
                                  wxf_view.shape[1],
                                  wxf_view.shape[2],
                                  wxf_view.shape[3]), dtype=wxf_view.dtype)
        t = 0
        while t < (self.nfac // h):
            filter_pair = numpy.concatenate(
                (
                    wxf_view[h*t:h*(t + 1), ...],
                    wyf_view[h*t:h*(t + 1), ...]
                ), 0)
            new_weights[h*2*t:h*2*(t + 1), ...] = filter_pair
            t += 1
        return new_weights

    def get_weights_view_shape(self):
        return (int(numpy.ceil(numpy.sqrt(self.nfac))),
                int(numpy.ceil(numpy.sqrt(self.nfac)))*2)


class DenoisingFactoredGatedAutoencoder(FactoredGatedAutoencoder):
    """
    A denoising gated autoencoder learns a relations between images by
    reconstructing a noisy version of them.

    Parameters
    ----------
    corruptor : object
        Instance of a corruptor object to use for corrupting the
        inputs.
    """
    def __init__(self, corruptor, nvisX, nvisY, nfac, nmap, recepF,
                 act_enc, act_dec, color=True, irange=1e-3, rng=9001):
        super(DenoisingFactoredGatedAutoencoder, self).__init__(
            nvisX,
            nvisY,
            nfac,
            nmap,
            recepF,
            act_enc,
            act_dec,
            color,
            irange,
            rng
            )
        self.corruptor = corruptor

    @wraps(FactoredGatedAutoencoder.reconstructXY)
    def reconstructXY(self, inputs):
        """
        Notes
        -----
        Reconstructions from corrupted data.
        """
        corrupted = self.corruptor(inputs)
        return (self.reconstructX(corrupted),
                self.reconstructY(corrupted))

    def reconstructXY_NoiseFree(self, inputs):
        return (self.reconstructX(inputs),
                self.reconstructY(inputs))
