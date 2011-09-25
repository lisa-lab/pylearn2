import numpy as np
from PIL import Image
import os
from pylearn2.utils import string
from tempfile import NamedTemporaryFile
import warnings

def show(image):
    """
    Parameters
    ----------
    image: A PIL Image, or a numpy ndarray.
            If ndarray, integer formats are assumed to use 0-255
            and float formats are assumed to use 0-1
    """

    if hasattr(image, '__array__'):
        if image.dtype == 'int8':
            image = np.cast['uint8'](image)
        elif str(image.dtype).startswith('float'):
            image *= 255.
            image = np.cast['uint8'](image)
        try:
            image = Image.fromarray(image)
        except TypeError:
            raise TypeError("PIL is whining about being given an ndarray of shape "+str(image.shape)+" and dtype "+str(image.dtype))

    try:
        f = NamedTemporaryFile(mode='r',suffix='.png', delete = False)
    except TypeError:
        #before python2.7, we can't use the delete argument
        f = NamedTemporaryFile(mode='r',suffix='.png')
        """
        TODO: prior to python 2.7, NamedTemporaryFile has no delete = False argument
            unfortunately, that means f.close() deletes the file.
            we then save an image to the file in the next line, so there's a race condition
            where for an instant we  don't actually have the file on the filesystem
            reserving the name, and then write to that name anyway
        """
        warnings.warn('filesystem race condition')

    name = f.name

    f.flush()
    f.close()

    image.save(name)

    viewer_command = string.preprocess('${PYLEARN2_VIEWER_COMMAND}')

    os.popen('('+viewer_command+' '+name+'; rm '+name+') &')

if __name__ == '__main__':
    black = np.zeros((50,50,3),dtype='uint8')

    red = black.copy()
    red[:,:,0] = 255

    green = black.copy()
    green[:,:,1] = 255

    show(black)
    show(green)
    show(red)
