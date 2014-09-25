"""
.. todo::

    WRITEME
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

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
    .. todo::

        WRITEME
    """

    def set_dbm(self, dbm):
        """
        .. todo::

            WRITEME
        """
        self.dbm = dbm

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError(str(type(self))+" does not implement mf.")

    def set_batch_size(self, batch_size):
        """
        If the inference procedure is dependent on a batch size at all, makes
        the necessary internal configurations to work with that batch size.
        """
        # TODO : was this supposed to be implemented?


class WeightDoubling(InferenceProcedure):
    """
    .. todo::

        WRITEME
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
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

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))

        #last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V)))

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            state_above = dbm.hidden_layers[-1].downward_state(Y)
            layer_above = dbm.hidden_layers[-1]
            assert len(dbm.hidden_layers) > 1

            # Last layer before Y does not need its weights doubled
            # because it already has top down input
            if len(dbm.hidden_layers) > 2:
                state_below = dbm.hidden_layers[-3].upward_state(H_hat[-3])
            else:
                state_below = dbm.visible_layer.upward_state(V)

            H_hat[-2] = dbm.hidden_layers[-2].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)

            # Last layer is clamped to Y
            H_hat[-1] = Y



        if block_grad == 1:
            H_hat = block(H_hat)

        history = [ list(H_hat) ]


        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(1, niter):
                for j in xrange(0,len(H_hat),2):
                    if j == 0:
                        state_below = dbm.visible_layer.upward_state(V)
                    else:
                        state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        layer_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)

                if Y is not None:
                    H_hat[-1] = Y

                for j in xrange(1,len(H_hat),2):
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        state_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                            state_below = state_below,
                            state_above = state_above,
                            layer_above = layer_above)
                    #end ifelse
                #end for odd layer

                if Y is not None:
                    H_hat[-1] = Y

                if block_grad == i:
                    H_hat = block(H_hat)

                history.append(list(H_hat))
            # end for mf iter
        # end if recurrent

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)
        if Y is not None:
            inferred = H_hat[:-1]
        else:
            inferred = H_hat
        for elem in flatten(inferred):
            # This check doesn't work with ('c', 0, 1, 'b') because 'b' is no longer axis 0
            # for value in get_debug_values(elem):
            #    assert value.shape[0] == dbm.batch_size
            assert V in gof.graph.ancestors([elem])
            if Y is not None:
                assert Y in gof.graph.ancestors([elem])
        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        if return_history:
            return history
        else:
            return H_hat


class SuperWeightDoubling(WeightDoubling):
    """
    .. todo::

        WRITEME
    """

    def multi_infer(self, V, return_history = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME
        """

        dbm = self.dbm

        assert return_history in [True, False, 0, 1]

        if niter is None:
            niter = dbm.niter

        new_V = 0.5 * V + 0.5 * dbm.visible_layer.init_inpainting_state(V,drop_mask = None,noise = False, return_unmasked = False)

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                                                            state_above = None,
                                                            double_weights = True,
                                                            state_below = dbm.visible_layer.upward_state(new_V),
                                                            iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                                                            state_above = None,
                                                            double_weights = True,
                                                            state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                                                            iter_name = '0'))

        #last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                                                         state_above = None,
                                                         state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                                                         state_above = None,
                                                         state_below = dbm.visible_layer.upward_state(V)))

        if block_grad == 1:
            H_hat = block(H_hat)

        history = [ (new_V, list(H_hat)) ]


        #we only need recurrent inference if there are multiple layers
        if len(H_hat) > 1:
            for i in xrange(1, niter):
                for j in xrange(0,len(H_hat),2):
                    if j == 0:
                        state_below = dbm.visible_layer.upward_state(new_V)
                    else:
                        state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        layer_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                                                              state_below = state_below,
                                                              state_above = state_above,
                                                              layer_above = layer_above)
                V_hat = dbm.visible_layer.inpaint_update(
                                                                                 state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                                                                                 layer_above = dbm.hidden_layers[0],
                                                                                 V = V,
                                                                                 drop_mask = None)
                new_V = 0.5 * V_hat + 0.5 * V

                for j in xrange(1,len(H_hat),2):
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                    if j == len(H_hat) - 1:
                        state_above = None
                        state_above = None
                    else:
                        state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                        layer_above = dbm.hidden_layers[j+1]
                    H_hat[j] = dbm.hidden_layers[j].mf_update(
                                                              state_below = state_below,
                                                              state_above = state_above,
                                                              layer_above = layer_above)
                #end ifelse
                #end for odd layer

                if block_grad == i:
                    H_hat = block(H_hat)
                    V_hat = block_gradient(V_hat)

                history.append((new_V, list(H_hat)))
        # end for mf iter
        # end if recurrent
        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)

        inferred = H_hat
        for elem in flatten(inferred):
            for value in get_debug_values(elem):
                assert value.shape[0] == dbm.batch_size
            assert V in gof.graph.ancestors([elem])

        if return_history:
            return history
        else:
            return H_hat[-1]

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME properly

        Gives the mean field expression for units masked out by drop_mask.
        Uses self.niter mean field updates.

        If you use this method in your research work, please cite:

            Multi-prediction deep Boltzmann machines. Ian J. Goodfellow,
            Mehdi Mirza, Aaron Courville, and Yoshua Bengio. NIPS 2013.


        Comes in two variants, unsupervised and supervised:

        * unsupervised: Y and drop_mask_Y are not passed to the method. The
          method produces V_hat, an inpainted version of V.
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

        dbm = self.dbm

        """TODO: Should add unit test that calling this with a batch of
                 different inputs should yield the same output for each
                 if noise is False and drop_mask is all 1s"""

        if niter is None:
            niter = dbm.niter

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
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
    "so each example is only one variable. drop_mask_Y should "
    "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V_hat),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))
        # Last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                #layer_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V_hat)))

        if Y is not None:
            Y_hat_unmasked = dbm.hidden_layers[-1].init_inpainting_state(Y, noise)
            dirty_term = drop_mask_Y * Y_hat_unmasked
            clean_term = (1 - drop_mask_Y) * Y
            Y_hat = dirty_term + clean_term
            H_hat[-1] = Y_hat
            if len(dbm.hidden_layers) > 1:
                i = len(dbm.hidden_layers) - 2
                if i == 0:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.visible_layer.upward_state(V_hat),
                        iter_name = '0')
                else:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                        iter_name = '0')


        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : list(H_hat), 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append(d)

        if block_grad == 1:
            V_hat = block_gradient(V_hat)
            V_hat_unmasked = block_gradient(V_hat_unmasked)
            H_hat = block(H_hat)
        update_history()

        for i in xrange(niter-1):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

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


class MoreConsistent(SuperWeightDoubling):
    """
    There's an oddity in SuperWeightDoubling where during the inpainting, we
    initialize Y_hat to sigmoid(biases) if a clean Y is passed in and 2 * weights
    otherwise. I believe but ought to check that mf always does weight doubling.
    This class makes the two more consistent by just implementing mf as calling
    inpainting with Y masked out.
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME
        """

        drop_mask = T.zeros_like(V)

        if Y is not None:
            # Y is observed, specify that it's fully observed
            drop_mask_Y = T.zeros_like(Y)
        else:
            # Y is not observed
            last_layer = self.dbm.hidden_layers[-1]
            if isinstance(last_layer, Softmax):
                # Y is not observed, the model has a Y variable, fill in a dummy one
                # and specify that no element of it is observed
                batch_size = self.dbm.get_input_space().batch_size(V)
                num_classes = self.dbm.hidden_layers[-1].n_classes
                assert isinstance(num_classes, int)
                Y = T.alloc(1., batch_size, num_classes)
                drop_mask_Y = T.alloc(1., batch_size)
            else:
                # Y is not observed because the model has no Y variable
                drop_mask_Y = None

        history = self.do_inpainting(V=V,
            Y=Y,
            return_history=True,
            drop_mask=drop_mask,
            drop_mask_Y=drop_mask_Y,
            noise=False,
            niter=niter,
            block_grad=block_grad)

        assert history[-1]['H_hat'][0] is not history[-2]['H_hat'][0] # rm

        if return_history:
            return [elem['H_hat'] for elem in history]

        rval =  history[-1]['H_hat']

        if 'Y_hat_unmasked' in history[-1]:
            rval[-1] = history[-1]['Y_hat_unmasked']

        return rval


class MoreConsistent2(WeightDoubling):
    """
    .. todo::

        WRITEME
    """

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        """
        .. todo::

            WRITEME properly

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

        dbm = self.dbm
        """TODO: Should add unit test that calling this with a batch of
                 different inputs should yield the same output for each
                 if noise is False and drop_mask is all 1s"""

        if niter is None:
            niter = dbm.niter

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
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
    "so each example is only one variable. drop_mask_Y should "
    "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            #do double weights update for_layer_i
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V_hat),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))
        # Last layer does not need its weights doubled, even on the first pass
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                #layer_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V_hat)))

        if Y is not None:
            Y_hat_unmasked = H_hat[-1]
            dirty_term = drop_mask_Y * Y_hat_unmasked
            clean_term = (1 - drop_mask_Y) * Y
            Y_hat = dirty_term + clean_term
            H_hat[-1] = Y_hat
            """
            if len(dbm.hidden_layers) > 1:
                i = len(dbm.hidden_layers) - 2
                if i == 0:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.visible_layer.upward_state(V_hat),
                        iter_name = '0')
                else:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                        iter_name = '0')
            """


        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : list(H_hat), 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append(d)

        if block_grad == 1:
            V_hat = block_gradient(V_hat)
            V_hat_unmasked = block_gradient(V_hat_unmasked)
            H_hat = block(H_hat)
        update_history()

        for i in xrange(niter-1):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

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


class BiasInit(InferenceProcedure):
    """
    An InferenceProcedure that initializes the mean field parameters based on the
    biases in the model. This InferenceProcedure uses the same weights at every
    iteration, rather than doubling the weights on the first pass.
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
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

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            # Last layer is clamped to Y
            H_hat[-1] = Y

        history = [ list(H_hat) ]

        #we only need recurrent inference if there are multiple layers
        assert (niter > 1) == (len(dbm.hidden_layers) > 1)

        for i in xrange(niter):
            for j in xrange(0,len(H_hat),2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)

            if Y is not None:
                H_hat[-1] = Y

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    state_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                #end ifelse
            #end for odd layer

            if Y is not None:
                H_hat[-1] = Y

            for i, elem in enumerate(H_hat):
                if elem is Y:
                    assert i == len(H_hat) -1
                    continue
                else:
                    assert elem not in history[-1]


            if block_grad == i + 1:
                H_hat = block(H_hat)

            history.append(list(H_hat))
        # end for mf iter

        # Run some checks on the output
        for layer, state in safe_izip(dbm.hidden_layers, H_hat):
            upward_state = layer.upward_state(state)
            layer.get_output_space().validate(upward_state)

        if Y is not None:
            assert H_hat[-1] is Y
            inferred = H_hat[:-1]
        else:
            inferred = H_hat
        for elem in flatten(inferred):
            for value in get_debug_values(elem):
                assert value.shape[0] == dbm.batch_size
            if V not in theano.gof.graph.ancestors([elem]):
                logger.error("{0} "
                             "does not have V as an ancestor!".format(elem))
                logger.error(theano.printing.min_informative_str(V))
                if elem is V:
                    logger.error("this variational parameter *is* V")
                else:
                    logger.error("this variational parameter "
                                 "is not the same as V")
                logger.error("V is {0}".format(V))
                assert False
            if Y is not None:
                assert Y in theano.gof.graph.ancestors([elem])

        if Y is not None:
            assert all([elem[-1] is Y for elem in history])
            assert H_hat[-1] is Y

        for elem in history:
            assert len(elem) == len(dbm.hidden_layers)

        if return_history:
            for hist_elem, H_elem in safe_zip(history[-1], H_hat):
                assert hist_elem is H_elem
            return history
        else:
            return H_hat

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
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

        dbm = self.dbm

        """TODO: Should add unit test that calling this with a batch of
                 different inputs should yield the same output for each
                 if noise is False and drop_mask is all 1s"""

        if niter is None:
            niter = dbm.niter


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
            assert isinstance(dbm.hidden_layers[-1], Softmax)
            if drop_mask_Y.ndim != 1:
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
                        "so each example is only one variable. drop_mask_Y should "
                        "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        if Y is not None:
            Y_hat_unmasked = dbm.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append( d )

        update_history()

        for i in xrange(niter):
            for j in xrange(0, len(H_hat), 2):
                if j == 0:
                    state_below = dbm.visible_layer.upward_state(V_hat)
                else:
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                    state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                    layer_above = dbm.hidden_layers[0],
                    V = V,
                    drop_mask = drop_mask, return_unmasked = True)
            V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            for j in xrange(1,len(H_hat),2):
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                #end if j
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(dbm.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y
                #end if y
            #end for j
            if block_grad == i + 1:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

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


class UpDown(InferenceProcedure):
    """
    An InferenceProcedure that initializes the mean field parameters based on the
    biases in the model, then alternates between updating each of the layers bottom-to-top
    and updating each of the layers top-to-bottom.
    """

    def mf(self, V, Y = None, return_history = False, niter = None, block_grad = None):
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

        H_hat = [None] + [layer.init_mf_state() for layer in dbm.hidden_layers[1:]]

        # Make corrections for if we're also running inference on Y
        if Y is not None:
            # Last layer is clamped to Y
            H_hat[-1] = Y

        history = [ list(H_hat) ]

        #we only need recurrent inference if there are multiple layers
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
                    state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = dbm.hidden_layers[j+1]
                H_hat[j] = dbm.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
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

    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
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
                raise ValueError("do_inpainting assumes Y is a matrix of one-hot labels,"
                        "so each example is only one variable. drop_mask_Y should "
                        "therefore be a vector, but we got something with ndim " +
                        str(drop_mask_Y.ndim))
            drop_mask_Y = drop_mask_Y.dimshuffle(0, 'x')

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = model.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = [None] + [layer.init_mf_state() for layer in model.hidden_layers[1:]]

        if Y is not None:
            Y_hat_unmasked = model.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            if Y is not None:
                d['Y_hat_unmasked'] = Y_hat_unmasked
                d['Y_hat'] = H_hat[-1]
            history.append( d )

        update_history()

        for i in xrange(niter):

            if i % 2 == 0:
                start = 0
                stop = len(H_hat)
                inc = 1
                if i > 0:
                    # Don't start by updating V_hat on iteration 0 or this will throw out the
                    # noise
                    V_hat, V_hat_unmasked = model.visible_layer.inpaint_update(
                            state_above = model.hidden_layers[0].downward_state(H_hat[0]),
                            layer_above = model.hidden_layers[0],
                            V = V,
                            drop_mask = drop_mask, return_unmasked = True)
                    V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)
            else:
                start = len(H_hat) - 1
                stop = -1
                inc = -1
            for j in xrange(start, stop, inc):
                if j == 0:
                    state_below = model.visible_layer.upward_state(V_hat)
                else:
                    state_below = model.hidden_layers[j-1].upward_state(H_hat[j-1])
                if j == len(H_hat) - 1:
                    state_above = None
                    layer_above = None
                else:
                    state_above = model.hidden_layers[j+1].downward_state(H_hat[j+1])
                    layer_above = model.hidden_layers[j+1]
                H_hat[j] = model.hidden_layers[j].mf_update(
                        state_below = state_below,
                        state_above = state_above,
                        layer_above = layer_above)
                if Y is not None and j == len(model.hidden_layers) - 1:
                    Y_hat_unmasked = H_hat[j]
                    H_hat[j] = drop_mask_Y * H_hat[j] + (1 - drop_mask_Y) * Y

            if i % 2 == 1:
                V_hat, V_hat_unmasked = model.visible_layer.inpaint_update(
                        state_above = model.hidden_layers[0].downward_state(H_hat[0]),
                        layer_above = model.hidden_layers[0],
                        V = V,
                        drop_mask = drop_mask, return_unmasked = True)
                V_hat.name = 'V_hat[%d](V_hat = %s)' % (i, V_hat.name)

            if block_grad == i + 1:
                V_hat = block_gradient(V_hat)
                V_hat_unmasked = block_gradient(V_hat_unmasked)
                H_hat = block(H_hat)
            update_history()
        #end for i

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
