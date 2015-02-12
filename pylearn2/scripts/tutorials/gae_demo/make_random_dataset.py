"""
This script creates a preprocessed dataset of image pairs
related by the defined transformation. The content of the
images is generated with a uniform distribution, this to
to show that the gating models do not depend on the
content but only on the relations.
"""

import itertools
import numpy
from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace


def generate(opc):
    """
    Summary (Generates a dataset with the chosen transformation).

    Parameters
    ----------
    opc: string
        Only two options, shifts or rotations.
    """
    dim = 19  # outer square
    # A bigger image is used to avoid empty pixels in the
    # borders.
    reg = 13  # inner square
    total = 20000  # Number of training examples

    im1 = numpy.zeros((total, reg, reg, 1), dtype='float32')
    im2 = numpy.zeros((total, reg, reg, 1), dtype='float32')
    Y = numpy.zeros((total, 1), dtype='uint8')
    rng = make_np_rng(9001, [1, 2, 3], which_method="uniform")
    transformation = opc

    if transformation == 'shifts':
        # Shifts
        # only shifts between [-3, +3] pixels
        shifts = list(itertools.product(range(-3, 4), range(-3, 4)))
        t = 0
        while t < total:
            x = rng.uniform(0, 1, (dim, dim))
            x = numpy.ceil(x * 255)
            im_x = x[3:16, 3:16][:, :, None]
            ind = rng.randint(0, len(shifts))
            Y[t] = ind
            txy = shifts[ind]
            tx, ty = txy
            im_y = x[(3 + tx):(16 + tx), (3 + ty):(16 + ty)][:, :, None]
            im1[t, :] = im_x
            im2[t, :] = im_y
            t += 1
    else:
        assert transformation == 'rotations'
        # Rotations
        import Image
        # import cv2
        angs = numpy.linspace(0, 359, 90)
        t = 0
        while t < total:
            x = rng.uniform(0, 1, (dim, dim))
            x = numpy.ceil(x * 255)
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

    # Normalize data:
    pipeline = preprocessing.Pipeline()
    gcn = preprocessing.GlobalContrastNormalization(
        sqrt_bias=10., use_std=True)
    pipeline.items.append(gcn)
    XY = numpy.concatenate((design_X, design_Y), 0)
    XY_ImP = dense_design_matrix.DenseDesignMatrix(X=XY)
    XY_ImP.apply_preprocessor(preprocessor=pipeline, can_fit=True)

    X1 = XY_ImP.X[0:design_X.shape[0], :]
    X2 = XY_ImP.X[design_X.shape[0]:, :]

    # As a Conv2DSpace
    topo_X1 = view_converter.design_mat_to_topo_view(X1)
    topo_X2 = view_converter.design_mat_to_topo_view(X2)
    axes = ('b', 0, 1, 'c')
    data_specs = (CompositeSpace(
        [Conv2DSpace((reg, reg), num_channels=1, axes=axes),
         Conv2DSpace((reg, reg), num_channels=1, axes=axes),
         VectorSpace(1)]),
        ('featuresX', 'featuresY', 'targets'))
    train = VectorSpacesDataset((topo_X1, topo_X2, Y), data_specs=data_specs)

    # As a VectorSpace
    # data_specs = (CompositeSpace(
    # [VectorSpace(reg * reg),
    # VectorSpace(reg * reg),
    #      VectorSpace(1)]),
    #               ('featuresX', 'featuresY', 'targets'))
    # train = VectorSpacesDataset(data=(X1, X2, Y), data_specs=data_specs)

    import os

    save_path = os.path.dirname(os.path.realpath(__file__))
    serial.save(os.path.join(save_path, 'train_preprocessed.pkl'), train)

if __name__ == '__main__':
    # Define the desired transformation between views
    generate('shifts')  # shifts or rotations
