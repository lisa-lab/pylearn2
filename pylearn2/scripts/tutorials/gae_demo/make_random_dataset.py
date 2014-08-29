# This script creates a preprocessed dataset of image pairs
# related by the defined transformation. The content of the
# images is generated with a uniform distribution, this to
# to show that the gating models do not depend on the
# content but only on the relations.

import itertools
import numpy
from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng


class ImagePairs(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, X, y=None):
        super(ImagePairs, self).__init__(X=X, y=y)


def generate(opc):
    # A bigger image is used to avoid empty pixels in the
    # borders.
    dim = 19  # outer square
    reg = 13  # inner square
    total = 20000  # Number of training examples

    im1 = numpy.zeros((total, reg, reg, 1), dtype='float32')
    im2 = numpy.zeros((total, reg, reg, 1), dtype='float32')
    Y = numpy.zeros((total, 1), dtype='uint8')
    rng = make_np_rng(9001, [1, 2, 3], which_method="uniform")
    transformation = opc

    if transformation == 'shifts':
        # Shifts
        print "Transformation:", transformation
        # only shifts between [-3, +3] pixels
        shifts = list(itertools.product(range(-3, 4), range(-3, 4)))
        t = 0
        while t < total:
            x = rng.uniform(0, 1, (dim, dim))
            x = numpy.ceil(x*255)
            im_x = x[3:16, 3:16][:, :, None]
            ind = rng.randint(0, len(shifts))
            Y[t] = ind
            txy = shifts[ind]
            tx, ty = txy
            im_y = x[(3+tx):(16+tx), (3+ty):(16+ty)][:, :, None]
            im1[t, :] = im_x
            im2[t, :] = im_y
            t += 1
    else:
        assert transformation == 'rotations'
        # Rotations
        print "Transformation:", transformation
        import Image
        # import cv2
        angs = numpy.linspace(0, 359, 90)
        t = 0
        while t < total:
            x = rng.uniform(0, 1, (dim, dim))
            x = numpy.ceil(x*255)
            im_x = x[3:16, 3:16][:, :, None]
            ind = rng.randint(0, len(angs))
            Y[t] = ind
            ang = angs[ind]
            y = numpy.asarray(Image.fromarray(x).rotate(ang))
            # scale = 1
            # M1 = cv2.getRotationMatrix2D((dim/2, dim/2), ang, scale)
            # y = cv2.warpAffine(x, M1, (dim, dim))
            im_y = y[3:16, 3:16][:, :, None]
            im1[t, :] = im_x
            im2[t, :] = im_y
            t += 1

    view_converter = dense_design_matrix.DefaultViewConverter((reg, reg, 1))
    design_X = view_converter.topo_view_to_design_mat(im1)
    design_Y = view_converter.topo_view_to_design_mat(im2)

    #NORMALIZE DATA:
    print "Normalizing"
    pipeline = preprocessing.Pipeline()
    gcn = preprocessing.GlobalContrastNormalization(
        sqrt_bias=10., use_std=True)
    pipeline.items.append(gcn)
    XY = numpy.concatenate((design_X, design_Y), 0)
    XY_ImP = ImagePairs(XY)
    XY_ImP.apply_preprocessor(preprocessor=pipeline, can_fit=True)

    X1 = XY_ImP.X[0:design_X.shape[0], :]
    X2 = XY_ImP.X[design_X.shape[0]:, :]
    X = numpy.concatenate((X1, X2), 1)

    print "Saving"
    train = ImagePairs(X=X, y=Y)
    import os
    save_path = os.path.dirname(os.path.realpath(__file__))
    train.use_design_loc(os.path.join(save_path, 'train_design.npy'))
    serial.save(os.path.join(save_path, 'train_preprocessed.pkl'), train)
    print "Done"

if __name__ == '__main__':
    # Define the desired transformation between views
    generate('shifts')  # shifts or rotations
