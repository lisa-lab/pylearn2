import sys
import numpy
from PIL import Image


def scale_to_unit_interval(ndar,eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / max(ndar.max(),eps)
    return ndar


def tile_raster_images(X, img_shape,
        tile_shape=None, tile_spacing=(1,1),
        scale_rows_to_unit_interval=True,
        output_pixel_vals=True,
        min_dynamic_range=1e-4,
        ):
    """
    Transform an array with one flattened image per row, into an array in which images are
    reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, and also columns of
    matrices for transforming those rows (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can be 2-D ndarrays or None
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols) (Defaults to a square-ish
        shape with the right area for the number of images)
    :type min_dynamic_range: positive float
    :param min_dynamic_range: the dynamic range of each image is used in scaling to the unit
        interval, but images with less dynamic range than this will be scaled as if this were
        the dynamic range.

    :returns: array suitable for viewing as an image.  (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
    # This is premature when tile_slices_to_image is not documented at all yet,
    # but ultimately true:
    #print >> sys.stderr, "WARN: tile_raster_images sucks, use tile_slices_to_image"
    if len(img_shape)==3 and img_shape[2]==3:
        # make this save an rgb image
        if scale_rows_to_unit_interval:
            print >> sys.stderr, "WARN: tile_raster_images' scaling routine messes up colour - try tile_slices_to_image"
        return tile_raster_images(
                (X[:,0::3], X[:,1::3], X[:,2::3], None),
                img_shape=img_shape[:2],
                tile_shape=tile_shape,
                tile_spacing=tile_spacing,
                scale_rows_to_unit_interval=scale_rows_to_unit_interval,
                output_pixel_vals=output_pixel_vals,
                min_dynamic_range=min_dynamic_range)

    if isinstance(X, tuple):
        n_images_in_x = X[0].shape[0]
    else:
        n_images_in_x = X.shape[0]

    if tile_shape is None:
        tile_shape = most_square_shape(n_images_in_x)

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    #out_shape is the shape in pixels of the returned image array
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        if scale_rows_to_unit_interval:
            raise NotImplementedError()
        assert len(X) == 4
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        )+channel_defaults[i]
            else:
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        H, W = img_shape
        Hs, Ws = tile_spacing

        out_scaling = 1
        if output_pixel_vals and str(X.dtype).startswith('float'):
            out_scaling = 255

        out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)
        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        try:
                            this_img = scale_to_unit_interval(
                                    X[tile_row * tile_shape[1] + tile_col].reshape(img_shape),
                                    eps=min_dynamic_range)
                        except ValueError:
                            raise ValueError('Failed to reshape array of shape %s to shape %s'
                                    % (
                                        X[tile_row*tile_shape[1] + tile_col].shape
                                        , img_shape
                                        ))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * out_scaling
        return out_array


def most_square_shape(N):
    """rectangle (height, width) with area N that is closest to sqaure
    """
    for i in xrange(int(numpy.sqrt(N)),0, -1):
        if 0 == N % i:
            return (i, N/i)


def save_tiled_raster_images(tiled_img, filename):
    """Save a a return value from `tile_raster_images` to `filename`.

    Returns the PIL image that was saved
    """
    if tiled_img.ndim==2:
        img = Image.fromarray( tiled_img, 'L')
    elif tiled_img.ndim==3:
        img = Image.fromarray(tiled_img, 'RGBA')
    else:
        raise TypeError('bad ndim', tiled_img)

    img.save(filename)
    return img


def tile_slices_to_image_uint8(X, tile_shape=None):
    if str(X.dtype) != 'uint8':
        raise TypeError(X)
    if tile_shape is None:
        #how many tile rows and cols
        (TR, TC) = most_square_shape(X.shape[0])
    H, W = X.shape[1], X.shape[2]

    Hs = H+1 #spacing between tiles
    Ws = W+1 #spacing between tiles

    trows, tcols= most_square_shape(X.shape[0])
    outrows = trows * Hs - 1
    outcols = tcols * Ws - 1
    out = numpy.zeros((outrows, outcols,3), dtype='uint8')
    tr_stride= 1+X.shape[1]
    for tr in range(trows):
        for tc in range(tcols):
            Xrc = X[tr*tcols+tc]
            if Xrc.ndim==2: # if no color channel make it broadcast
                Xrc=Xrc[:,:,None]
            #print Xrc.shape
            #print out[tr*Hs:tr*Hs+H,tc*Ws:tc*Ws+W].shape
            out[tr*Hs:tr*Hs+H,tc*Ws:tc*Ws+W] = Xrc
    img = Image.fromarray(out, 'RGB')
    return img


def tile_slices_to_image(X,
        tile_shape=None,
        scale_each=True,
        min_dynamic_range=1e-4):
    #always returns an RGB image
    def scale_0_255(x):
        xmin = x.min()
        xmax = x.max()
        return numpy.asarray(
                255 * (x - xmin) / max(xmax - xmin, min_dynamic_range),
                dtype='uint8')

    if scale_each:
        uintX = numpy.empty(X.shape, dtype='uint8')
        for i, Xi in enumerate(X):
            uintX[i] = scale_0_255(Xi)
        X = uintX
    else:
        X = scale_0_255(X)
    return tile_slices_to_image_uint8(X, tile_shape=tile_shape)
