import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.axes
import os
from pylearn2.utils import string_utils as string
from tempfile import NamedTemporaryFile
import warnings


def imview(*args, **kwargs):
    """
    A more sensible matplotlib-based image viewer command,
    a wrapper around `matplotlib.pyplot.imshow`.

    Parameters are identical to `matplotlib.pyplot.imshow`
    but this behaves somewhat differently:

      * By default, it creates a new figure (unless a
        `figure` keyword argument is supplied.
      * It modifies the axes of that figure to use the
        full frame, without ticks or tick labels.
      * It turns on `nearest` interpolation by default
        (i.e., it does not antialias pixel data). This
        can be overridden with the `interpolation`
        argument as in `imshow`.

    All other arguments and keyword arguments are passed
    on to `imshow`.`
    """
    if 'figure' not in kwargs:
        f = plt.figure()
    else:
        f = kwargs['figure']
    new_ax = matplotlib.axes.Axes(f, [0, 0, 1, 1],
                                  xticks=[], yticks=[],
                                  frame_on=False)
    f.delaxes(f.gca())
    f.add_axes(new_ax)
    if len(args) < 5 and 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    plt.imshow(*args, **kwargs)


def show(image):
    """
    Parameters
    ----------
    image : PIL Image object or ndarray
        If ndarray, integer formats are assumed to use 0-255
        and float formats are assumed to use 0-1
    """
    if hasattr(image, '__array__'):
        #do some shape checking because PIL just raises a tuple indexing error
        #that doesn't make it very clear what the problem is
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError('image must have either 2 or 3 dimensions but its shape is '+str(image.shape))

        if image.dtype == 'int8':
            image = np.cast['uint8'](image)
        elif str(image.dtype).startswith('float'):
            #don't use *=, we don't want to modify the input array
            image = image * 255.
            image = np.cast['uint8'](image)

        #PIL is too stupid to handle single-channel arrays
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:,:,0]

        try:
            image = Image.fromarray(image)
        except TypeError:
            raise TypeError("PIL issued TypeError on ndarray of shape " +
                            str(image.shape) + " and dtype " +
                            str(image.dtype))


    try:
        f = NamedTemporaryFile(mode='r', suffix='.png', delete=False)
    except TypeError:
        # before python2.7, we can't use the delete argument
        f = NamedTemporaryFile(mode='r', suffix='.png')
        """
        TODO: prior to python 2.7, NamedTemporaryFile has no delete = False
        argument unfortunately, that means f.close() deletes the file.  we then
        save an image to the file in the next line, so there's a race condition
        where for an instant we  don't actually have the file on the filesystem
        reserving the name, and then write to that name anyway

        TODO: see if this can be remedied with lower level calls (mkstemp)
        """
        warnings.warn('filesystem race condition')

    name = f.name
    f.flush()
    f.close()
    image.save(name)
    viewer_command = string.preprocess('${PYLEARN2_VIEWER_COMMAND}')
    os.popen('(' + viewer_command + ' ' + name + '; rm ' + name + ') &')


def pil_from_ndarray(ndarray):
    try:
        if ndarray.dtype == 'float32' or ndarray.dtype == 'float64':
            assert ndarray.min() >= 0.0
            assert ndarray.max() <= 1.0

            ndarray = np.cast['uint8'](ndarray * 255)

            if len(ndarray.shape) == 3 and ndarray.shape[2] == 1:
                ndarray = ndarray[:, :, 0]

        rval = Image.fromarray(ndarray)
        return rval
    except Exception, e:
        raise
        print 'original exception: '
        print e
        print 'ndarray.dtype: ', ndarray.dtype
        print 'ndarray.shape: ', ndarray.shape

    assert False


def ndarray_from_pil(pil, dtype='uint8'):

    rval = np.asarray(pil)

    if dtype != rval.dtype:
        rval = np.cast[dtype](rval)

    if str(dtype).startswith('float'):
        rval /= 255.

    if len(rval.shape) == 2:
        rval = rval.reshape(rval.shape[0], rval.shape[1], 1)

    return rval


def rescale(image, shape):
    """ scales image to be no larger than shape
        PIL might give you unexpected results beyond that"""

    assert len(image.shape) == 3  # rows, cols, channels
    assert len(shape) == 2  # rows, cols

    i = pil_from_ndarray(image)

    i.thumbnail([shape[1], shape[0]], Image.ANTIALIAS)

    rval = ndarray_from_pil(i, dtype=image.dtype)

    return rval
resize = rescale

def fit_inside(image, shape):
    """ scales image down to fit inside shape
        preserves proportions of image"""

    assert len(image.shape) == 3  # rows, cols, channels
    assert len(shape) == 2  # rows, cols

    if image.shape[0] <= shape[0] and image.shape[1] <= shape[1]:
        return image.copy()

    row_ratio = float(image.shape[0]) / float(shape[0])
    col_ratio = float(image.shape[1]) / float(shape[1])

    if row_ratio > col_ratio:
        target_shape = [shape[0], min(image.shape[1] / row_ratio, shape[1])]
    else:
        target_shape = [min(image.shape[0] / col_ratio, shape[0]), shape[1]]

    assert target_shape[0] <= shape[0]
    assert target_shape[1] <= shape[1]
    assert target_shape[0] == shape[0] or target_shape[1] == shape[1]
    rval = rescale(image, target_shape)
    return rval


def letterbox(image, shape):
    """ pads image with black letterboxing to bring image.shape up to shape """

    assert len(image.shape) == 3  # rows, cols, channels
    assert len(shape) == 2  # rows, cols

    assert image.shape[0] <= shape[0]
    assert image.shape[1] <= shape[1]

    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image.copy()

    rval = np.zeros((shape[0], shape[1], image.shape[2]), dtype=image.dtype)

    rstart = (shape[0] - image.shape[0]) / 2
    cstart = (shape[1] - image.shape[1]) / 2

    rend = rstart + image.shape[0]
    cend = cstart + image.shape[1]
    rval[rstart:rend, cstart:cend] = image

    return rval


def make_letterboxed_thumbnail(image, shape):
    """
    scales image down to shape
    preserves proportions of image, introduces black letterboxing if necessary
    """

    assert len(image.shape) == 3
    assert len(shape) == 2

    shrunk = fit_inside(image, shape)
    letterboxed = letterbox(shrunk, shape)

    return letterboxed


def load(filepath, rescale=True, dtype='float64'):
    assert type(filepath) == str

    if rescale == False and dtype == 'uint8':
        rval = np.asarray(Image.open(filepath))
        # print 'image.load: ' + str((rval.min(), rval.max()))
        assert rval.dtype == 'uint8'
        return rval

    s = 1.0
    if rescale:
        s = 255.
    try:
        rval = Image.open(filepath)
    except:
        raise Exception("Could not open "+filepath)

    rval = np.cast[dtype](np.asarray(rval)) / s

    if len(rval.shape) == 2:
        rval = rval.reshape(rval.shape[0], rval.shape[1], 1)

    if len(rval.shape) != 3:
        raise AssertionError("Something went wrong opening " +
                filepath + '. Resulting shape is ' + str(rval.shape) +
                " (it's meant to have 3 dimensions by now)")

    return rval

def save(filepath, ndarray):
    pil_from_ndarray(ndarray).save(filepath)

if __name__ == '__main__':
    black = np.zeros((50, 50, 3), dtype='uint8')

    red = black.copy()
    red[:, :, 0] = 255

    green = black.copy()
    green[:, :, 1] = 255

    show(black)
    show(green)
    show(red)
