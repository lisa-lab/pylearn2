"""
.. todo::

    WRITEME
"""
import logging

from theano.compat.six.moves import input

from pylearn2.utils import serial
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse
from pylearn2.datasets import control
import numpy as np


from pylearn2.utils.exc import reraise_as


logger = logging.getLogger(__name__)


def get_weights_report(model_path=None,
                       model=None,
                       rescale='individual',
                       border=False,
                       norm_sort=False,
                       dataset=None):
    """
    Returns a PatchViewer displaying a grid of filter weights

    Parameters
    ----------
    model_path : str
        Filepath of the model to make the report on.
    rescale : str
        A string specifying how to rescale the filter images:
            - 'individual' (default) : scale each filter so that it
                  uses as much as possible of the dynamic range
                  of the display under the constraint that 0
                  is gray and no value gets clipped
            - 'global' : scale the whole ensemble of weights
            - 'none' :   don't rescale
    dataset : pylearn2.datasets.dataset.Dataset
        Dataset object to do view conversion for displaying the weights. If
        not provided one will be loaded from the model's dataset_yaml_src.

    Returns
    -------
    WRITEME
    """

    if model is None:
        logger.info('making weights report')
        logger.info('loading model')
        model = serial.load(model_path)
        logger.info('loading done')
    else:
        assert model_path is None
    assert model is not None

    if rescale == 'none':
        global_rescale = False
        patch_rescale = False
    elif rescale == 'global':
        global_rescale = True
        patch_rescale = False
    elif rescale == 'individual':
        global_rescale = False
        patch_rescale = True
    else:
        raise ValueError('rescale=' + rescale +
                         ", must be 'none', 'global', or 'individual'")


    if isinstance(model, dict):
        #assume this was a saved matlab dictionary
        del model['__version__']
        del model['__header__']
        del model['__globals__']
        keys = [key for key in model \
                if hasattr(model[key], 'ndim') and model[key].ndim == 2]
        if len(keys) > 2:
            key = None
            while key not in keys:
                logger.info('Which is the weights?')
                for key in keys:
                    logger.info('\t{0}'.format(key))
                key = input()
        else:
            key, = keys
        weights = model[key]

        norms = np.sqrt(np.square(weights).sum(axis=1))
        logger.info('min norm: {0}'.format(norms.min()))
        logger.info('mean norm: {0}'.format(norms.mean()))
        logger.info('max norm: {0}'.format(norms.max()))

        return patch_viewer.make_viewer(weights,
                                        is_color=weights.shape[1] % 3 == 0)

    weights_view = None
    W = None

    try:
        weights_view = model.get_weights_topo()
        h = weights_view.shape[0]
    except NotImplementedError:

        if dataset is None:
            logger.info('loading dataset...')
            control.push_load_data(False)
            dataset = yaml_parse.load(model.dataset_yaml_src)
            control.pop_load_data()
            logger.info('...done')

        try:
            W = model.get_weights()
        except AttributeError as e:
            reraise_as(AttributeError("""
Encountered an AttributeError while trying to call get_weights on a model.
This probably means you need to implement get_weights for this model class,
but look at the original exception to be sure.
If this is an older model class, it may have weights stored as weightsShared,
etc.
Original exception: """+str(e)))

    if W is None and weights_view is None:
        raise ValueError("model doesn't support any weights interfaces")

    if weights_view is None:
        weights_format = model.get_weights_format()
        assert hasattr(weights_format,'__iter__')
        assert len(weights_format) == 2
        assert weights_format[0] in ['v','h']
        assert weights_format[1] in ['v','h']
        assert weights_format[0] != weights_format[1]

        if weights_format[0] == 'v':
            W = W.T
        h = W.shape[0]

        if norm_sort:
            norms = np.sqrt(1e-8+np.square(W).sum(axis=1))
            norm_prop = norms / norms.max()


        weights_view = dataset.get_weights_view(W)
        assert weights_view.shape[0] == h
    try:
        hr, hc = model.get_weights_view_shape()
    except NotImplementedError:
        hr = int(np.ceil(np.sqrt(h)))
        hc = hr
        if 'hidShape' in dir(model):
            hr, hc = model.hidShape

    pv = patch_viewer.PatchViewer(grid_shape=(hr, hc),
                                  patch_shape=weights_view.shape[1:3],
            is_color = weights_view.shape[-1] == 3)

    if global_rescale:
        weights_view /= np.abs(weights_view).max()

    if norm_sort:
        logger.info('sorting weights by decreasing norm')
        idx = sorted( range(h), key=lambda l : - norm_prop[l] )
    else:
        idx = range(h)

    if border:
        act = 0
    else:
        act = None

    for i in range(0,h):
        patch = weights_view[idx[i],...]
        pv.add_patch(patch, rescale=patch_rescale, activation=act)

    abs_weights = np.abs(weights_view)
    logger.info('smallest enc weight magnitude: {0}'.format(abs_weights.min()))
    logger.info('mean enc weight magnitude: {0}'.format(abs_weights.mean()))
    logger.info('max enc weight magnitude: {0}'.format(abs_weights.max()))


    if W is not None:
        norms = np.sqrt(np.square(W).sum(axis=1))
        assert norms.shape == (h,)
        logger.info('min norm: {0}'.format(norms.min()))
        logger.info('mean norm: {0}'.format(norms.mean()))
        logger.info('max norm: {0}'.format(norms.max()))

    return pv


def get_binocular_greyscale_weights_report(model_path=None,
                                           model=None,
                                           rescale='individual',
                                           border=False,
                                           norm_sort=False,
                                           dataset=None):
    """
    Returns a PatchViewer displaying a grid of filter weights

    Parameters
    ----------
    model_path : str
        Filepath of the model to make the report on.
    rescale : str
        A string specifying how to rescale the filter images:

          - 'individual' (default) : scale each filter so that it
            uses as much as possible of the dynamic range
            of the display under the constraint that 0
            is gray and no value gets clipped
          - 'global' : scale the whole ensemble of weights
          - 'none' : don't rescale
    dataset : pylearn2.datasets.dataset.Dataset
        Dataset object to do view conversion for displaying the weights. If \
        not provided one will be loaded from the model's dataset_yaml_src.

    Returns
    -------
    WRITEME
    """

    if model is None:
        logger.info('making weights report')
        logger.info('loading model')
        model = serial.load(model_path)
        logger.info('loading done')
    else:
        assert model_path is None
    assert model is not None

    if rescale == 'none':
        global_rescale = False
        patch_rescale = False
    elif rescale == 'global':
        global_rescale = True
        patch_rescale = False
    elif rescale == 'individual':
        global_rescale = False
        patch_rescale = True
    else:
        raise ValueError('rescale=' + rescale +
                         ", must be 'none', 'global', or 'individual'")


    if isinstance(model, dict):
        #assume this was a saved matlab dictionary
        del model['__version__']
        del model['__header__']
        del model['__globals__']
        weights ,= model.values()

        norms = np.sqrt(np.square(weights).sum(axis=1))
        logger.info('min norm: {0}'.format(norms.min()))
        logger.info('mean norm: {0}'.format(norms.mean()))
        logger.info('max norm: {0}'.format(norms.max()))

        return patch_viewer.make_viewer(weights,
                                        is_color=weights.shape[1] % 3 == 0)

    weights_view = None
    W = None

    try:
        weights_view = model.get_weights_topo()
        h = weights_view.shape[0]
    except NotImplementedError:

        if dataset is None:
            logger.info('loading dataset...')
            control.push_load_data(False)
            dataset = yaml_parse.load(model.dataset_yaml_src)
            control.pop_load_data()
            logger.info('...done')

        try:
            W = model.get_weights()
        except AttributeError as e:
            reraise_as(AttributeError("""
Encountered an AttributeError while trying to call get_weights on a model.
This probably means you need to implement get_weights for this model class,
but look at the original exception to be sure.
If this is an older model class, it may have weights stored as weightsShared,
etc.
Original exception: """+str(e)))

    if W is None and weights_view is None:
        raise ValueError("model doesn't support any weights interfaces")

    if weights_view is None:
        if hasattr(model,'get_weights_format'):
            weights_format = model.get_weights_format()
        elif hasattr(model, 'weights_format'):
            weights_format = model.weights_format
        else:
            # assume default
            weights_format = ('v', 'h')


        assert hasattr(weights_format,'__iter__')
        assert len(weights_format) == 2
        assert weights_format[0] in ['v','h']
        assert weights_format[1] in ['v','h']
        assert weights_format[0] != weights_format[1]

        if weights_format[0] == 'v':
            W = W.T
        h = W.shape[0]

        if norm_sort:
            norms = np.sqrt(1e-8+np.square(W).sum(axis=1))
            norm_prop = norms / norms.max()


        weights_view = dataset.get_weights_view(W)
        assert weights_view.shape[0] == h
    try:
        hr, hc = model.get_weights_view_shape()
    except NotImplementedError:
        hr = int(np.ceil(np.sqrt(h)))
        hc = hr
        if 'hidShape' in dir(model):
            hr, hc = model.hidShape

    pv = patch_viewer.PatchViewer(grid_shape=(hr, hc * 2),
                                  patch_shape=weights_view.shape[1:3],
                                  is_color=weights_view.shape[-1] == 3)

    if global_rescale:
        weights_view /= np.abs(weights_view).max()

    if norm_sort:
        logger.info('sorting weights by decreasing norm')
        idx = sorted(range(h), key=lambda l : - norm_prop[l])
    else:
        idx = range(h)

    if border:
        act = 0
    else:
        act = None

    for i in range(0,h):
        patch = weights_view[idx[i],...]
        if patch_rescale:
            patch = patch / np.abs(patch).max()
        pv.add_patch(patch[:,:,1], rescale=False, activation=act)
        pv.add_patch(patch[:,:,0], rescale=False, activation=act)

    abs_weights = np.abs(weights_view)
    logger.info('smallest enc weight magnitude: {0}'.format(abs_weights.min()))
    logger.info('mean enc weight magnitude: {0}'.format(abs_weights.mean()))
    logger.info('max enc weight magnitude: {0}'.format(abs_weights.max()))


    if W is not None:
        norms = np.sqrt(np.square(W).sum(axis=1))
        assert norms.shape == (h,)
        logger.info('min norm: {0}'.format(norms.min()))
        logger.info('mean norm: {0}'.format(norms.mean()))
        logger.info('max norm: {0}'.format(norms.max()))

    return pv
