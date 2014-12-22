"""
Factored Gated AutoEncoder Class.

This code is adapted from the implementation of Roland Memisevic,
specifically from the paper:
"Gradient-based learning of higher-order image features"
(http://www.iro.umontreal.ca/~memisevr/code/rae/index.html).
And the autoencoder implementation of Pylearn2.
"""

__authors__ = "Raul Peralta Lozada"
__copyright__ = "(c) 2014, The University of Edinburgh"
__license__ = "3-clause BSD License"
__contact__ = "raulpl25@gmail.com"

# Third-party imports
import numpy
from theano import tensor

# Local imports
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng, make_theano_rng
from pylearn2.models import Model
from pylearn2.blocks import Block
from pylearn2.utils import wraps
from pylearn2.utils.data_specs import DataSpecsMapping
import theano

SMALL = 0.000001


class GatedAutoencoder(Block, Model):
    """
    Base class for Gated Autoencoder. You should not use this class
    directly, use one of its subclasses. This model receives a pair of
    datasets, like image pairs. You can use VectorSpacesDataset to specify
    your data.

    Parameters
    ----------
    nmap : int
        Number of mappings units.
    input_space : Space object, opt
        A CompositeSpace specifying the kind of inputs. If None, input space
        is specified by nvisX and nvisY.
    nvisx : int, opt
        Number of visible units in the first set.
    nvisy : int, opt
        Number of visible units in the second set.
    input_source : tuple of strings, opt
        A tuple of strings specifiying the input sources this
        model accepts. The structure should match that of input_space.
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
    irange : float, opt
        Width of the initial range around 0 from which to sample initial
        values for the weights.
    rng : RandomState object or seed, opt
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """

    def __init__(self, nmap, input_space=None, nvisx=None, nvisy=None,
                 input_source=("featuresX", "featuresY"),
                 act_enc=None, act_dec=None,
                 irange=1e-3, rng=9001):
        Block.__init__(self)
        Model.__init__(self)
        assert nmap > 0, "Number of mapping units must be positive"

        if nvisx is not None and nvisy is not None or input_space is not None:
            if nvisx is not None and nvisy is not None:
                assert nvisx > 0, "Number of visx units must be non-negative"
                assert nvisy > 0, "Number of visy units must be non-negative"
                input_space = CompositeSpace([
                    VectorSpace(nvisx),
                    VectorSpace(nvisy)])
                self.nvisx = nvisx
                self.nvisy = nvisy
            elif isinstance(input_space.components[0], Conv2DSpace):
                rx, cx = input_space.components[0].shape
                chx = input_space.components[0].num_channels
                ry, cy = input_space.components[1].shape
                chy = input_space.components[1].num_channels
                self.nvisx = rx * cx * chx
                self.nvisy = ry * cy * chy
            else:
                raise NotImplementedError(
                    str(type(self)) + " does not support that input_space.")
        # Check whether the input_space and input_source structures match
        try:
            DataSpecsMapping((input_space, input_source))
        except ValueError:
            raise ValueError("The structures of `input_space`, %s, and "
                             "`input_source`, %s do not match. If you "
                             "specified a CompositeSpace as an input, "
                             "be sure to specify the data sources as well."
                             % (input_space, input_source))

        self.input_space = input_space
        self.input_source = input_source
        self.nmap = nmap
        self.output_space = VectorSpace(self.nmap)
        self._initialize_visbiasX(self.nvisx)  # self.visbiasX
        self._initialize_visbiasY(self.nvisy)  # self.visbiasY
        self._initialize_mapbias()  # self.mapbias
        self.irange = irange
        self.rng = make_np_rng(rng, which_method="randn")
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

    def _initialize_visbiasX(self, nvisx):
        """
        Initialize the biases of the first set of visible units.
        """
        self.visbiasX = sharedX(
            numpy.zeros(nvisx),
            name='vbX',
            borrow=True
        )

    def _initialize_visbiasY(self, nvisy):
        """
        Initialize the biases of the second set of visible units.
        """
        self.visbiasY = sharedX(
            numpy.zeros(nvisy),
            name='vbY',
            borrow=True
        )

    get_input_space = Model.get_input_space
    get_output_space = Model.get_output_space
    get_input_source = Model.get_input_source


class FactoredGatedAutoencoder(GatedAutoencoder):
    """
    Factored Gated Autoencoder model.

    Parameters
    ----------
    nfac : int
        Number of factors used to project the images.
    nmap : int
        Number of mappings units.
    input_space : Space object, opt
        A Space specifying the kind of input. If None, input space
        is specified by nvisX and nvisY.
    nvisX : int, opt
        Number of visible units in the first image.
    nvisY : int, opt
        Number of visible units in the second image.
    input_source : tuple of strings, opt
        A tuple of strings specifiying the input sources this
        model accepts. The structure should match that of input_space.
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
    irange : float, opt
        Width of the initial range around 0 from which to sample initial
        values for the weights.
    rng : RandomState object or seed, opt
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """

    def __init__(self, nfac, nmap, input_space=None, nvisx=None, nvisy=None,
                 input_source=("featuresX", "featuresY"),
                 act_enc=None, act_dec=None,
                 irange=1e-3, rng=9001):
        super(FactoredGatedAutoencoder, self).__init__(
            nmap,
            input_space,
            nvisx,
            nvisy,
            input_source,
            act_enc,
            act_dec,
            irange,
            rng,
        )

        # Parameters
        self.nfac = nfac
        self._initialize_wxf(self.nvisx, self.nfac)  # self.wxf
        self._initialize_wyf(self.nvisy, self.nfac)  # self.wyf
        self._initialize_whf(self.nmap, self.nfac)  # self.whf
        self._initialize_whf_in(self.nmap, self.nfac)  # self.whf_in

        self._params = [self.wxf, self.wyf, self.whf_in, self.whf,
                        self.mapbias, self.visbiasX, self.visbiasY]

    def _initialize_wxf(self, nvisx, nfac, rng=None, irange=None):
        """
        Creation of weight matrix wxf.
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.wxf = sharedX(
            (rng.randn(nvisx, nfac)) * irange,
            name='wxf',
            borrow=True
        )

    def _initialize_wyf(self, nvisy, nfac, rng=None, irange=None):
        """
        Creation of weight matrix wyf.
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.wyf = sharedX(
            (rng.randn(nvisy, nfac)) * irange,
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
        """
        Applies the filters wxf to the first input and returns
        the corresponding factors
        """
        return tensor.dot(inputs[0], self.wxf)

    def _factorsY(self, inputs):
        """
        Applies the filters wyf to the second input and returns
        the corresponding factors
        """
        return tensor.dot(inputs[1], self.wyf)

    def _mappings(self, inputs):
        """
        Returns the mapping units.
        """
        return self.mapbias + tensor.dot(
            self._factorsX(inputs) * self._factorsY(inputs), self.whf_in.T)

    def _hidden_activation(self, inputs):
        """
        Single minibatch activation function.

        Parameters
        ----------
        inputs : tensor_like
            Theano symbolic representing the input minibatch
            that consists of a tuple of spaces with sizes (nviX, nvisY).

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

    def _factorsH(self, inputs):
        """
        Returns the factors corresponding to the mapping units.
        """
        return tensor.dot(self._hidden_activation(inputs), self.whf)

    def decodeX(self, inputs):
        """
        Returns the reconstruction of 'x' before the act_dec function

        Parameters
        ----------
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).
        """
        return self.visbiasX + tensor.dot(
            self._factorsY(inputs) * self._factorsH(inputs), self.wxf.T)

    def decodeY(self, inputs):
        """
        Returns the reconstruction of 'y' before the act_dec function

        Parameters
        ----------
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).
        """
        return self.visbiasY + tensor.dot(
            self._factorsX(inputs) * self._factorsH(inputs), self.wyf.T)

    def reconstructX(self, inputs):
        """
        Reconstruction of input x.

        Parameters
        ----------
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).
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
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).
        """
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        return act_dec(self.decodeY(inputs))

    def reconstructXY(self, inputs):
        """
        Reconstruction of both datasets.

        Parameters
        ----------
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).

        Returns
        -------
        Reconstruction: tuple
            Tuple (lenght 2) of the tensor_like reconstruction of the
            datasets.
        """
        return (self.reconstructX(inputs),
                self.reconstructY(inputs))

    def __call__(self, inputs):
        """
        This just aliases the `_hidden_activation()` function for syntactic
        sugar/convenience.
        """
        return self._hidden_activation(inputs)

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
        nwxf = (wxf_updated.std(0) + SMALL)[numpy.newaxis, :]
        nwyf = (wyf_updated.std(0) + SMALL)[numpy.newaxis, :]
        meannxf = nwxf.mean()
        meannyf = nwyf.mean()
        # Center filters
        centered_wxf = wxf_updated - wxf_updated.mean(0)
        centered_wyf = wyf_updated - wyf_updated.mean(0)
        # Fix standard deviation
        wxf_updated = centered_wxf * (meannxf / nwxf)
        wyf_updated = centered_wyf * (meannyf / nwyf)
        updates[wxf] = wxf_updated
        updates[wyf] = wyf_updated

    def get_weights_topo(self):
        """
        Returns a topological view of the weights, the first half
        corresponds to wxf and the second half to wyf.

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
        if (not isinstance(self.input_space.components[0], Conv2DSpace) or
                not isinstance(self.input_space.components[1], Conv2DSpace)):
            raise NotImplementedError()
        wxf = self.wxf.get_value(borrow=False).T
        wyf = self.wyf.get_value(borrow=False).T
        convx = self.input_space.components[0]
        convy = self.input_space.components[1]
        vecx = VectorSpace(self.nvisx)
        vecy = VectorSpace(self.nvisy)
        wxf_view = vecx.np_format_as(wxf,
                                     Conv2DSpace(
                                         convx.shape,
                                         num_channels=convx.num_channels,
                                         axes=('b', 0, 1, 'c')))
        wyf_view = vecy.np_format_as(wyf,
                                     Conv2DSpace(
                                         convy.shape,
                                         num_channels=convy.num_channels,
                                         axes=('b', 0, 1, 'c')))
        h = int(numpy.ceil(numpy.sqrt(self.nfac)))
        new_weights = numpy.zeros((wxf_view.shape[0] * 2,
                                   wxf_view.shape[1],
                                   wxf_view.shape[2],
                                   wxf_view.shape[3]), dtype=wxf_view.dtype)
        t = 0
        while t < (self.nfac // h):
            filter_pair = numpy.concatenate(
                (
                    wxf_view[h * t:h * (t + 1), ...],
                    wyf_view[h * t:h * (t + 1), ...]
                ), 0)
            new_weights[h * 2 * t:h * 2 * (t + 1), ...] = filter_pair
            t += 1
        return new_weights

    @wraps(GatedAutoencoder.get_weights_view_shape)
    def get_weights_view_shape(self):
        return (int(numpy.ceil(numpy.sqrt(self.nfac))),
                int(numpy.ceil(numpy.sqrt(self.nfac))) * 2)


class DenoisingFactoredGatedAutoencoder(FactoredGatedAutoencoder):
    """
    A denoising gated autoencoder learns a relations between images by
    reconstructing a noisy version of them.

    Parameters
    ----------
    corruptor : object
        Instance of a corruptor object to use for corrupting the
        inputs.
    nfac : int
        Number of factors used to project the images.
    nmap : int
        Number of mappings units.
    input_space : Space object, opt
        A Space specifying the kind of input. If None, input space
        is specified by nvisX and nvisY.
    nvisX : int, opt
        Number of visible units in the first image.
    nvisY : int, opt
        Number of visible units in the second image.
    input_source : tuple of strings, opt
        A tuple of strings specifiying the input sources this
        model accepts. The structure should match that of input_space.
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
    irange : float, opt
        Width of the initial range around 0 from which to sample initial
        values for the weights.
    rng : RandomState object or seed, opt
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """

    def __init__(self, corruptor, nfac, nmap, input_space=None,
                 nvisx=None, nvisy=None,
                 input_source=("featuresX", "featuresY"),
                 act_enc=None, act_dec=None, irange=1e-3, rng=9001):
        super(DenoisingFactoredGatedAutoencoder, self).__init__(
            nfac,
            nmap,
            input_space,
            nvisx,
            nvisy,
            input_source,
            act_enc,
            act_dec,
            irange,
            rng
        )
        self.corruptor = corruptor

    def reconstructXY(self, inputs):
        """
        Reconstruction of both datasets.

        Parameters
        ----------
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).

        Returns
        -------
        Reconstruction: tuple
            Tuple (lenght 2) of the tensor_like reconstruction of the
            datasets.

        Notes
        -----
        Reconstructions from corrupted data.
        """
        corrupted = self.corruptor(inputs)
        return (self.reconstructX(corrupted),
                self.reconstructY(corrupted))

    def reconstructXY_NoiseFree(self, inputs):
        """
        Method that returns the reconstruction without noise

        Parameters
        ----------
        inputs : tuple
            Tuple (lenght 2) of theano symbolic  representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing the two data dimensions (X, Y).

        Returns
        -------
        Reconstruction: tuple
            Tuple (lenght 2) of the tensor_like reconstruction of the
            datasets.
        """
        return (self.reconstructX(inputs),
                self.reconstructY(inputs))
