from pylearn2.utils import serial
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse
from pylearn2.datasets import control
import numpy as np
import warnings

def get_weights_report(model_path = None, model = None, rescale = 'individual', border = False, norm_sort = False,
        dataset = None):
    """
        Returns a PatchViewer displaying a grid of filter weights

        Parameters:
            model_path: the filepath of the model to make the report on.
            rescale: a string specifying how to rescale the filter images
                        'individual' (default): scale each filter so that it
                            uses as much as possible of the dynamic range
                            of the display under the constraint that 0
                            is gray and no value gets clipped
                        'global' : scale the whole ensemble of weights
                        'none' :   don't rescale
            dataset: a Dataset object to do view conversion for displaying the weights.
                    if not provided one will be loaded from the model's dataset_yaml_src
    """

    if model is None:
        print 'making weights report'
        print 'loading model'
        model = serial.load(model_path)
        print 'loading done'
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
        raise ValueError('rescale='+rescale+", must be 'none', 'global', or 'individual'")


    if isinstance(model, dict):
        #assume this was a saved matlab dictionary
        del model['__version__']
        del model['__header__']
        del model['__globals__']
        weights ,= model.values()

        norms = np.sqrt(np.square(weights).sum(axis=1))
        print 'min norm: ',norms.min()
        print 'mean norm: ',norms.mean()
        print 'max norm: ',norms.max()

        return patch_viewer.make_viewer(weights, is_color = weights.shape[1] % 3 == 0)

    weights_view = None
    W = None

    try:
        weights_view = model.get_weights_topo()
        h = weights_view.shape[0]
    except Exception, e:

        if dataset is None:
            print 'loading dataset...'
            control.push_load_data(False)
            dataset = yaml_parse.load(model.dataset_yaml_src)
            control.pop_load_data()
            print '...done'

        try:
            W = model.get_weights()
        except AttributeError, e:
            raise AttributeError("""
Encountered an AttributeError while trying to call get_weights on a model.
This probably means you need to implement get_weights for this model class,
but look at the original exception to be sure.
If this is an older model class, it may have weights stored as weightsShared,
etc.
Original exception: """+str(e))

        has_D = False
        if 'D' in dir(model):
            has_D = True
            D = model.D

        if 'enc_weights_shared' in dir(model):
            W = model.enc_weights_shared.get_value()


        if W is None:
            raise AttributeError('model does not have a variable with a name like "W", "weights", etc  that pylearn2 recognizes')


    if (W is not None and len(W.shape) == 2) or weights_view is not None:
        if weights_view is None:
            if hasattr(model,'get_weights_format'):
                weights_format = model.get_weights_format()
            if hasattr(model, 'weights_format'):
                weights_format = model.weights_format

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
        #print 'weights_view shape '+str(weights_view.shape)
        hr = int(np.ceil(np.sqrt(h)))
        hc = hr
        if 'hidShape' in dir(model):
            hr, hc = model.hidShape

        pv = patch_viewer.PatchViewer(grid_shape=(hr,hc), patch_shape=weights_view.shape[1:3],
                is_color = weights_view.shape[-1] == 3)

        if global_rescale:
            weights_view /= np.abs(weights_view).max()

        if norm_sort:
            print 'sorting weights by decreasing norm'
            idx = sorted( range(h), key = lambda l : - norm_prop[l] )
        else:
            idx = range(h)

        if border:
            act = 0
        else:
            act = None

        for i in range(0,h):
            patch = weights_view[idx[i],...]
            pv.add_patch( patch, rescale   = patch_rescale, activation = act)
    else:
        e = model.weights
        d = model.dec_weights_shared.value

        h = e.shape[0]

        if len(e.shape) == 8:
            raise Exception("get_weights_report doesn't support tiled convolution yet, use the show_weights8 app")

        if e.shape[4] != 1:
            raise Exception('weights shape: '+str(e.shape))
        shape = e.shape[1:3]
        dur = e.shape[3]

        show_dec = id(e) != id(d)

        pv = patch_viewer.PatchViewer( grid_shape = ((1+show_dec)*h,dur), patch_shape=shape)
        for i in range(0,h):
            pv.addVid( e[i,:,:,:,0], rescale = rescale)
            if show_dec:
                pv.addVid( d[i,:,:,:,0], rescale = rescale)

    print 'smallest enc weight magnitude: '+str(np.abs(weights_view).min())
    print 'mean enc weight magnitude: '+str(np.abs(weights_view).mean())
    print 'max enc weight magnitude: '+str(np.abs(weights_view).max())


    if W is not None:
        norms = np.sqrt(np.square(W).sum(axis=1))
        assert norms.shape == (h,)
        print 'min norm: ',norms.min()
        print 'mean norm: ',norms.mean()
        print 'max norm: ',norms.max()

    return pv
