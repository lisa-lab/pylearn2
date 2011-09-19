import numpy as np
from PIL import Image
import os
from tempfile import NamedTemporaryFile

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

    f = NamedTemporaryFile(mode='r',suffix='.png',delete=False)

    name = f.name

    f.flush()
    f.close()


    image.save(name)

    os.popen('(eog --new-instance '+name+'; rm '+name+') &')


if __name__ == '__main__':
    black = np.zeros((50,50,3),dtype='uint8')

    red = black.copy()
    red[:,:,0] = 255

    green = black.copy()
    green[:,:,1] = 255

    show(black)
    show(green)
    show(red)
