"""
Various InferenceProcedures for use with the DBM class.
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin", "Devon Hjelm"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import functools
import logging

from theano import gof
import theano.tensor as T
import theano
from theano.gof.op import get_debug_values

from pylearn2.models.dbm import block, flatten
from pylearn2.models.dbm.layer import Softmax
from pylearn2.utils import safe_izip, block_gradient, safe_zip


logger = logging.getLogger(__name__)


class InferenceProcedure(object):

    """
    A class representing a procedure for performing mean field inference in a
    DBM.
    Different subclasses can implement different specific procedures, such as
    updating the layers in different orders, or using different strategies to
    initialize the mean field expectations.
    """

    def set_dbm(self, dbm):
        """
        Associates the InferenceProcedure with a specific DBM.

        Parameters
        ----------
        dbm : pylearn2.models.dbm.DBM instance
            The model to perform inference in.
        """
        self.dbm = dbm

    def mf(self, V, Y=None, return_history=False, niter=None, block_grad=None):
        """
        Perform mean field inference. Subclasses must implement.

        Parameters
        ----------
        V : Input space batch
            The values of the input features modeled by the DBM.
        Y : (Optional) Target space batch
            The values of the labels modeled by the DBM. Must be omitted
            if the DBM does not model labels. If the DBM does model
            labels, they may be included to perform inference over the
            hidden layers only, or included to perform inference over the
            labels.
        return_history : (Optional) bool
            Default: False
            If True, returns the full sequence of mean field updates.
        niter : (Optional) int
        block_grad : (Optional) int
            Default: None
            If not None, blocks the gradient after `block_grad`
            iterations, so that only the last `niter` - `block_grad`
            iterations need to be stored when using the backpropagation
            algorithm.

        Returns
        -------
        result : list
            If not `return_history` (default), a list with one element
            per inferred layer, containing the full mean field state
            of that layer.
            Otherwise, a list of such lists, with the outer list
            containing one element for each step of inference.
        """
        raise NotImplementedError(str(type(self)) + " does not implement mf.")

    def set_batch_size(self, batch_size):
        """
        If the inference procedure is dependent on a batch size at all, makes
        the necessary internal configurations to work with that batch size.

        Parameters
        ----------
        batch_size : int
            The number of examples in the batch
        """
        # Default implementation is no-op, because default procedure does
        # not depend on the batch size.

    def multi_infer(self, V, return_history=False, niter=None,
                    block_grad=None):
        """
        Inference using "the multi-inference trick." See
        "Multi-prediction deep Boltzmann machines", Goodfellow et al 2013.

        Subclasses may implement this method, however it is not needed for
        any training algorithm, and only expected to work at evaluation
        time if the model was trained with multi-prediction training.

        Parameters
        ----------
        V : input space batch
        return_history : bool
            If True, returns the complete history of the mean field
            iterations, rather than just the final values
        niter : int
            The number of mean field iterations to run
        block_grad : int
            If not None, block the gradient after this number of iterations

        Returns
        -------
        result : list
            A list of mean field states, or if return_history is True, a
            list of such lists with one element per mean field iteration
        """

        raise NotImplementedError(str(type(self)) + " does not implement"
                                  " multi_infer.")

    def do_inpainting(self, V, Y=None, drop_mask=None, drop_mask_Y=None,
                      return_history=False, noise=False, niter=None,
                      block_grad=None):
        """
        Does the inference required for multi-prediction training.

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V
        * supervised: Y and drop_mask_Y are passed to the method. The method
          produces V_hat and Y_hat

        Parameters
        ----------
        V : tensor_like
            Theano batch in `model.input_space`
        Y : tensor_like
            Theano batch in `model.output_space`, i.e. in the output space of
            the last hidden layer. (It's not really a hidden layer anymore,
            but oh well. It's convenient to code it this way because the
            labels are sort of "on top" of everything else.) *** Y is always
            assumed to be a matrix of one-hot category labels. ***
        drop_mask : tensor_like
            Theano batch in `model.input_space`. Should be all binary, with
            1s indicating that the corresponding element of X should be
            "dropped", i.e. hidden from the algorithm and filled in as part
            of the inpainting process
        drop_mask_Y : tensor_like
            Theano vector. Since we assume Y is a one-hot matrix, each row is
            a single categorical variable. `drop_mask_Y` is a binary mask
            specifying which *rows* to drop.
        return_history : bool, optional
            WRITEME
        noise : bool, optional
            WRITEME
        niter : int, optional
            WRITEME
        block_grad : WRITEME

        Returns
        -------
        WRITEME
        """

        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "do_inpainting.")

    def is_rbm_compatible(self):
        """
        Checks whether inference procedure is compatible with an RBM.

        A restricted Boltzmann machine (RBM) is a deep Boltzmann machine (DBM)
        with exactly one hidden layer. Inference of the posterior is exactly
        equivalent to one mean field update of the hidden units given the data.
        An rbm compatible inference procedure should:
        1) calculate the posterior of the hidden units from the data as
        defined by the joint probability P(v,h) = 1/Z e^E(v,h), where E(.) is
        the energy over the graph and Z is the marginal.
        2) not involve cross terms between hidden units.
        3) not double or replicate weights.
        4) use exactly one mean field step.
        """

        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "is_rbm_compatible.")


class UpDown(InferenceProcedure):

    """
    An InferenceProcedure that initializes the mean field parameters
    based on the biases in the model, then alternates between updating
    each of the layers bottom-to-top
    and updating each of the layers top-to-bottom.
    """

    @functools.wraps(InferenceProcedure.mf)
    def mf(self, V, Y=None, return_history=False, niter=None, block_grad=None):
        """
        .. todo::

            WRITEME
        """

        dbm = self.dbm

        assert Y not in [True, False, 0, 1]
        assert return_history in [True, False, 0, 1]

        if Y is not None:
            dbm.hidden_layers[-1].get_output_space().validate(Y)

        if niter is None:
            niter = dbm.niter

        H_hat = [None] + [layer.init_mf_state()
                          for layer in dbm.hidden_layers[1:]]

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            # Last layer is clamped to Y
            H_hat[-1] = Y

        history = [list(H_hat)]

        # we only need recurrent inference if there are multiple layers
        assert (niter > 1) == (len(dbm.hidden_layers) > 1)

        for i in xrange(niter):
            # Determine whether to go up or down on this iteration
            if i % 2 == 0:
                start = 0
                stop = len(H_hat)
                inc = 1
            else:
                start = len(H_hat) - 1
                stop = -1
                inc = -1
            # Do the mean field updates
            for j in xrange(start, stop, inc):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V)
                else:
                    state_below = dbm.hidden_layers[
                        j - 1].upward_state(H_hat[j - 1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[
                        j + 1].downward_state(H_hat[j + 1])
                    layer_above = dbm.hidden_layers[j + 1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                    state_below=state_below,
                    state_above=state_above,
                    layer_above=layer_above)
                if Y is not None:
                    H_hat[-1] = Y

            if Y is not None:
                H_hat[-1] = Y

            if block_grad == i + 1:
                H_hat = block(H_hat)

            history.append(list(H_hat))
        # end for mf iter

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)
        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        if return_history:
            return history
        else:
            return H_hat

    def do_inpainting(self, V, Y=None, drop_mask=None, drop_mask_Y=None,
                      return_history=False, noise=False, niter=None,
                      block_grad=None):
        """
        .. todo::

            WRITEME properly

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V.
        * supervised: Y and drop_mask_Y are passed to the method. The method
          produces V_hat and Y_hat.

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Parameters
        ----------
        V : tensor_like
            Theano batch in `model.input_space`
        Y : tensor_like
            Theano batch in model.output_space, ie, in the output space of
            the last hidden layer (it's not really a hidden layer anymore,
            but oh well. It's convenient to code it this way because the
            labels are sort of "on top" of everything else). *** Y is always
            assumed to be a matrix of one-hot category labels. ***
        drop_mask : tensor_like
            A theano batch in `model.input_space`. Should be all binary, with
            1s indicating that the corresponding element of X should be
            "dropped", ie, hidden from the algorithm and filled in as part of
            the inpainting process
        drop_mask_Y : tensor_like
            Theano vector. Since we assume Y is a one-hot matrix, each row is
            a single categorical variable. `drop_mask_Y` is a binary mask
            specifying which *rows* to drop.
        """

        if Y is not None:
            assert isinstance(self.hidden_layers[-1], Softmax)

        model = self.dbm

        """TODO: Should add unit test that calling this with a batch of
                 different inputs should yield the same output for each
                 if noise is False and drop_mask is all 1s"""

        if niter is None:
            niter = model.niter

        assert drop_mask is not None
        assert return_history in [True, False]
        assert noise in [True, False]
        if Y is None:
            if drop_mask_Y is not None:
                raise ValueError("do_inpainting got drop_mask_Y but not Y.")
        else:
            if drop_mask_Y is None:
                raise ValueError("do_inpainting got Y but not drop_mask_Y.")

        if Y is not None:
            assert isinstance(model.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of"
                                 "one-hot labels,"
                                 "so each example is only one variable. "
                                 "drop_mask_Y should "
                                 "therefore be a vector, but we got "
                                 "something with ndim " +
                                 str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = model.visible_layer.init_inpainting_state(
            V, drop_mask, noise, return_unmasked=True)
        assert V_hat_unmasked.ndim > 1

        H_hat = [None] + [layer.init_mf_state()
                          for layer in model.hidden_layers[1:]]

        if Y is not None:
            Y_hat_unmasked = model.hidden_layers[
                -1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d = {'V_hat':  V_hat, 'H_hat': H_hat,
                 'V_hat_unmasked': V_hat_unmasked}
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append(d)

        update_history()

        for i in xrange(niter):

            if i % 2 == 0:
                start = 0
                stop = len(H_hat)
                inc = 1
                if i > 0:
                    # Don't start by updating V_hat on iteration 0 or
                    # this will throw out the noise
                    V_hat, V_hat_unmasked = model.visible_layer.inpaint_update(
                        state_above=model.hidden_layers[0].downward_state(
                            H_hat[0]),
                        layer_above=model.hidden_layers[0],
                        V=V,
                        drop_mask=drop_mask, return_unmasked=True)
                    V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)
            else:
                start = len(H_hat) - 1
                stop = -1
                inc = -1
            for j in xrange(start, stop, inc):
                if j == 0:
                    state_below = model.visible_layer.upward_state(V_hat)
                else:
                    state_below = model.hidden_layers[
                        j - 1].upward_state(H_hat[j - 1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = model.hidden_layers[
                        j + 1].downward_state(H_hat[j + 1])
                    layer_above = model.hidden_layers[j + 1]
                H_hat[j] = model.hidden_layers[j].mf_update(
                    state_below=state_below,
                    state_above=state_above,
                    layer_above=layer_above)
                if Y is not None and j == len(model.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            if i % 2 == 1:
                V_hat, V_hat_unmasked = model.visible_layer.inpaint_update(
                    state_above=model.hidden_layers[0].downward_state(
                        H_hat[0]),
                    layer_above=model.hidden_layers[0],
                    V=V,
                    drop_mask=drop_mask, return_unmasked=True)
                V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            if block_grad == i + 1:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        # end for i

        # debugging, make sure V didn't get changed in this function
        assert V is orig_V
        assert drop_mask is orig_drop_mask

        Y_hat = H_hat[-1]

        assert V in theano.gof.graph.ancestors([V_hat])
        if Y is not None:
            assert V in theano.gof.graph.ancestors([Y_hat])

        if return_history:
            return history
        else:
            if Y is not None:
                return V_hat, Y_hat
            return V_hat

    def is_rbm_compatible(self):
        """
        Is implemented as UpDown is RBM compatible.
        """
        return