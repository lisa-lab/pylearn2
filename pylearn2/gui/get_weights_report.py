from framework.utils import serial
from framework.gui import patch_viewer
from framework.config import yaml_parse
import numpy as N

def get_weights_report(model_path, rescale = True):
    print 'making weights report'
    print 'loading model'
    p = serial.load(model_path)
    print 'loading done'

    dataset = yaml_parse.load(p.dataset_yaml_src)

    if 'weightsShared' in dir(p):
        p.weights = p.weightsShared.get_value()

    if 'W' in dir(p):
        p.weights = p.W.get_value()

    if 'D' in dir(p):
        p.decWeightsShared = p.D

    if 'enc_weights_shared' in dir(p):
        p.weights = p.enc_weights_shared.get_value()


    if 'W' in dir(p) and len(p.W.get_value().shape) == 3:
        W = p.W.get_value()
        nh , nv, ns = W.shape
        pv = patch_viewer.PatchViewer(grid_shape=(nh,ns), patch_shape= dataset.view_shape()[0:2])

        for i in range(0,nh):
            for k in range(0,ns):
                patch = W[i,:,k]
                patch = dataset.vec_to_view(patch, weights = True)
                pv.add_patch( patch, rescale = rescale)
            #
        #
    elif len(p.weights.shape) == 2:
        assert type(p.weights_format()) == type([])
        assert len(p.weights_format()) == 2
        assert p.weights_format()[0] in ['v','h']
        assert p.weights_format()[1] in ['v','h']
        assert p.weights_format()[0] != p.weights_format()[1]

        if p.weights_format()[0] == 'v':
            p.weights = p.weights.transpose()
        h = p.nhid


        hr = int(N.ceil(N.sqrt(h)))
        hc = hr
        if 'hidShape' in dir(p):
            hr, hc = p.hidShape

        pv = patch_viewer.PatchViewer(grid_shape=(hr,hc), patch_shape=dataset.view_shape()[0:2],
                is_color = dataset.view_shape()[2] == 3)
        weights_mat = p.weights

        assert weights_mat.shape[0] == h
        weights_view = dataset.get_weights_view(weights_mat)
        assert weights_view.shape[0] == h
        #print 'weights_view shape '+str(weights_view.shape)
        for i in range(0,h):
            patch = weights_view[i,...]
            pv.add_patch( patch, rescale   = rescale)
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

        pv = PatchViewer.PatchViewer( grid_shape = ((1+show_dec)*h,dur), patch_shape=shape)
        for i in range(0,h):
            pv.addVid( e[i,:,:,:,0], rescale = rescale)
            if show_dec:
                pv.addVid( d[i,:,:,:,0], rescale = rescale)

    print 'smallest enc weight magnitude: '+str(N.abs(p.weights).min())
    print 'mean enc weight magnitude: '+str(N.abs(p.weights).mean())
    print 'max enc weight magnitude: '+str(N.abs(p.weights).max())


    return pv
