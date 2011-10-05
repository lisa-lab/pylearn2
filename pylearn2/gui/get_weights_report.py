from pylearn2.utils import serial
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse
import numpy as np
import warnings

def get_weights_report(model_path, rescale = 'individual'):
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
    """

    print 'making weights report'
    print 'loading model'
    p = serial.load(model_path)
    print 'loading done'

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

    dataset = yaml_parse.load(p.dataset_yaml_src)

    if hasattr(p,'get_weights'):
        W = p.get_weights()

    if 'weightsShared' in dir(p):
        W = p.weightsShared.get_value()

    if 'W' in dir(p):
        if hasattr(p.W,'__array__'):
            warnings.warn('model.W is an ndarray; I can figure out how to display this but that seems like a sign of a bad bug')
            W = p.W
        else:
            W = p.W.get_value()

    has_D = False
    if 'D' in dir(p):
        has_D = True
        D = p.D

    if 'enc_weights_shared' in dir(p):
        W = p.enc_weights_shared.get_value()


    if len(W.shape) == 2:
        if hasattr(p,'get_weights_format'):
            weights_format = p.get_weights_format()
        if hasattr(p, 'weights_format'):
            weights_format = p.weights_format

        assert hasattr(weights_format,'__iter__')
        assert len(weights_format) == 2
        assert weights_format[0] in ['v','h']
        assert weights_format[1] in ['v','h']
        assert weights_format[0] != weights_format[1]

        if weights_format[0] == 'v':
            W = W.T
        h = W.shape[0]


        norms = np.sqrt(1e-8+np.square(W).sum(axis=1))

        norm_prop = norms / norms.max()

        hr = int(np.ceil(np.sqrt(h)))
        hc = hr
        if 'hidShape' in dir(p):
            hr, hc = p.hidShape

        pv = patch_viewer.PatchViewer(grid_shape=(hr,hc), patch_shape=dataset.view_shape()[0:2],
                is_color = dataset.view_shape()[2] == 3)

        weights_view = dataset.get_weights_view(W)
        assert weights_view.shape[0] == h
        #print 'weights_view shape '+str(weights_view.shape)

        if global_rescale:
            weights_view /= np.abs(weights_view).max()


        print 'sorting weights by decreasing norm'
        idx = sorted( range(h), key = lambda l : - norm_prop[l] )

        for i in range(0,h):
            patch = weights_view[idx[i],...]
            pv.add_patch( patch, rescale   = patch_rescale)#, activation = norm_prop[idx[i]])
    else:
        e = p.weights
        d = p.dec_weights_shared.value

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

    print 'smallest enc weight magnitude: '+str(np.abs(W).min())
    print 'mean enc weight magnitude: '+str(np.abs(W).mean())
    print 'max enc weight magnitude: '+str(np.abs(W).max())


    norms = np.sqrt(np.square(W).sum(axis=1))
    assert norms.shape == (h,)
    print 'min norm: ',norms.min()
    print 'mean norm: ',norms.mean()
    print 'max norm: ',norms.max()

    return pv
