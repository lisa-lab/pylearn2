"""
Script to generate the MNIST+ dataset. The purpose of this dataset is to make a
more challenging MNIST-like dataset, with multiple factors of variation. These
factors can serve to evaluate a model's performance at learning invariant
features, or its ability to disentangle factors of variation in a multi-task
classification setting. The dataset is stored under $PYLEARN2_DATA_PATH.

The dataset variants are created as follows. For each MNIST image, we:
1. Perform a random rotation of the image (optional)
2. Rescale the image from 28x28 to 48x48, yielding variable `image`.
3.1 Extract a random patch `textured_patch` from a fixed or random image of the
Brodatz texture dataset.
3.2 Generate mask of MNIST digit outline, by thresholding MNIST digit at 0.1
3.3 Fuse MNIST digit and textured patch as follows:
    textured_patch[mask] <= image[mask]; image <= textured_patch;
4. Randomly select position of light source (optional)
5. Perform embossing operation, given fixed lighting position obtained in 4.
"""
import numpy
from theano.compat.six.moves import xrange, cPickle as pickle
import pylab as pl

from copy import copy
from optparse import OptionParser

from pylearn2.datasets import mnist
from pylearn2.utils import string_utils

import warnings
try:
    from PIL import Image
except ImportError:
    warnings.warn("Couldn't import Image from PIL, so far make_mnistplus "
            "is only supported with PIL")


OUTPUT_SIZE = 48
DOWN_SAMPLE = 1


def to_array(img):
    """
    Convert PIL.Image to numpy.ndarray.
    :param img: numpy.ndarray
    """
    return numpy.array(img.getdata()) / 255.


def to_img(arr, os):
    """
    Convert numpy.ndarray to PIL.Image
    :param arr: numpy.ndarray
    :param os: integer, size of output image.
    """
    return Image.fromarray(arr.reshape(os, os) * 255.)


def emboss(img, azi=45., ele=18., dep=2):
    """
    Perform embossing of image `img`.
    :param img: numpy.ndarray, matrix representing image to emboss.
    :param azi: azimuth (in degrees)
    :param ele: elevation (in degrees)
    :param dep: depth, (0-100)
    """
    # defining azimuth, elevation, and depth
    ele = (ele * 2 * numpy.pi) / 360.
    azi = (azi * 2 * numpy.pi) / 360.

    a = numpy.asarray(img).astype('float')
    # find the gradient
    grad = numpy.gradient(a)
    # (it is two arrays: grad_x and grad_y)
    grad_x, grad_y = grad
    # getting the unit incident ray
    gd = numpy.cos(ele) # length of projection of ray on ground plane
    dx = gd * numpy.cos(azi)
    dy = gd * numpy.sin(azi)
    dz = numpy.sin(ele)
    # adjusting the gradient by the "depth" factor
    # (I think this is how GIMP defines it)
    grad_x = grad_x * dep / 100.
    grad_y = grad_y * dep / 100.
    # finding the unit normal vectors for the image
    leng = numpy.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x/leng
    uni_y = grad_y/leng
    uni_z = 1./leng
    # take the dot product
    a2 = 255 * (dx*uni_x + dy*uni_y + dz*uni_z)
    # avoid overflow
    a2 = a2.clip(0, 255)
    # you must convert back to uint8 /before/ converting to an image
    return Image.fromarray(a2.astype('uint8'))


def extract_patch(textid, os, downsample):
    """
    Extract a patch of texture #textid of Brodatz dataset.
    :param textid: id of texture image to load.
    :param os: size of MNIST+ output images.
    :param downsample: integer, downsampling factor.
    """
    temp = '${PYLEARN2_DATA_PATH}/textures/brodatz/D%i.gif' % textid
    fname = string_utils.preprocess(temp)

    img_i = Image.open(fname)
    img_i = img_i.resize((img_i.size[0]/downsample,
                          img_i.size[1]/downsample), Image.BILINEAR)

    x = numpy.random.randint(0, img_i.size[0] - os)
    y = numpy.random.randint(0, img_i.size[1] - os)
    patch = img_i.crop((x, y, x+os, y+os))

    return patch, (x, y)


def gendata(enable, os, downsample, textid=None, seed=2313, verbose=False):
    """
    Generate the MNIST+ dataset.
    :param enable: dictionary of flags with keys ['texture', 'azimuth',
    'rotation', 'elevation'] to enable/disable a given factor of variation.
    :param textid: if enable['texture'], id number of the Brodatz texture to
    load. If textid is None, we load a random texture for each MNIST image.
    :param os: output size (width and height) of MNIST+ images.
    :param downsample: factor by which to downsample texture.
    :param seed: integer for seeding RNG.
    :param verbose: bool
    """
    rng = numpy.random.RandomState(seed)

    data  = mnist.MNIST('train')
    test  = mnist.MNIST('test')
    data.X = numpy.vstack((data.X, test.X))
    data.y = numpy.hstack((data.y, test.y))
    del test

    output = {}
    output['data']  = numpy.zeros((len(data.X), os*os))
    output['label'] = numpy.zeros(len(data.y))
    if enable['azimuth']:
        output['azimuth'] = numpy.zeros(len(data.y))
    if enable['elevation']:
        output['elevation'] = numpy.zeros(len(data.y))
    if enable['rotation']:
        output['rotation'] = numpy.zeros(len(data.y))
    if enable['texture']:
        output['texture_id']  = numpy.zeros(len(data.y))
        output['texture_pos'] = numpy.zeros((len(data.y), 2))

    for i in xrange(len(data.X)):

        # get MNIST image
        frgd_img = to_img(data.X[i], 28)
        frgd_img = frgd_img.convert('L')

        if enable['rotation']:
            rot = rng.randint(0, 360)
            output['rotation'][i] = rot
            frgd_img = frgd_img.rotate(rot, Image.BILINEAR)

        frgd_img = frgd_img.resize((os, os), Image.BILINEAR)

        if enable['texture']:

            if textid is None:
                # extract patch from texture database. Note that texture #14
                # does not exist.
                textid = 14
                while textid == 14:
                    textid = rng.randint(1, 113)

            patch_img, (px, py) = extract_patch(textid, os, downsample)
            patch_arr = to_array(patch_img)

            # store output details
            output['texture_id'][i] = textid
            output['texture_pos'][i] = (px, py)

            # generate binary mask for digit outline
            frgd_arr = to_array(frgd_img)
            mask_arr = frgd_arr > 0.1

            # copy contents of masked-MNIST image into background texture
            blend_arr = copy(patch_arr)
            blend_arr[mask_arr] = frgd_arr[mask_arr]

            # this now because the image to emboss
            frgd_img = to_img(blend_arr, os)

        azi = 45
        if enable['azimuth']:
            azi = rng.randint(0, 360)
            output['azimuth'][i] = azi
        ele = 18.
        if enable['elevation']:
            ele = rng.randint(0, 60)
            output['elevation'][i] = ele

        mboss_img = emboss(frgd_img, azi=azi, ele=ele)
        mboss_arr = to_array(mboss_img)

        output['data'][i] = mboss_arr
        output['label'][i] = data.y[i]

        if verbose:
            pl.imshow(mboss_arr.reshape(os, os))
            pl.gray()
            pl.show()

    fname = 'mnistplus'
    if enable['azimuth']:
        fname += "_azi"
    if enable['rotation']:
        fname += "_rot"
    if enable['texture']:
        fname += "_tex"
    fp = open(fname+'.pkl','w')
    pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', action='store_true', dest='verbose')
    parser.add_option('--azimuth', action='store_true', dest='azimuth',
            help='Enable random azimuth for light-source used in embossing.')
    parser.add_option('--elevation', action='store_true', dest='elevation',
            help='Enable random elevation for light-source used in embossing.')
    parser.add_option('--rotation', action='store_true', dest='rotation',
            help='Randomly rotate MNIST digit prior to embossing.')
    parser.add_option('--texture', action='store_true', dest='texture',
            help='Perform joint embossing of fused {MNIST + Texture} image.')
    parser.add_option('--textid', action='store', type='int', dest='textid',
            help='If specified, use a single texture ID for all MNIST images.',
            default=None)
    parser.add_option('--output_size', action='store', type='int', dest='os',
            help='Integer specifying size of (square) output images.',
            default=OUTPUT_SIZE)
    parser.add_option('--downsample', action='store', type='int',
            dest='downsample', default=DOWN_SAMPLE,
            help='Downsampling factor for Brodatz textures.')
    (opts, args) = parser.parse_args()

    enable = {'texture':   opts.texture,
              'azimuth':   opts.azimuth,
              'rotation':  opts.rotation,
              'elevation': opts.elevation}

    gendata(enable=enable, os=opts.os, downsample=opts.downsample,
            verbose=opts.verbose, textid=opts.textid)
