import numpy as np
from PIL import Image
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils.image import show
from pylearn2.utils import image
import warnings

def make_viewer(mat, grid_shape=None, patch_shape=None, activation=None, pad=None, is_color = False, rescale = True):
    """ Given filters in rows, guesses dimensions of patches
        and nice dimensions for the PatchViewer and returns a PatchViewer
        containing visualizations of the filters"""

    num_channels = 1
    if is_color:
        num_channels = 3

    if grid_shape is None:
        grid_shape = PatchViewer.pick_shape(mat.shape[0] )
    if mat.ndim > 2:
        patch_shape = mat.shape[1:3]
        topo_view = mat
        num_channels = mat.shape[3]
        is_color = num_channels > 1
    else:
        if patch_shape is None:
            assert mat.shape[1] % num_channels == 0
            patch_shape = PatchViewer.pick_shape(mat.shape[1] / num_channels, exact = True)
            assert patch_shape[0] * patch_shape[1] * num_channels == mat.shape[1]
        topo_shape = (patch_shape[0], patch_shape[1], num_channels)
        view_converter = DefaultViewConverter(topo_shape)
        topo_view = view_converter.design_mat_to_topo_view(mat)
    rval = PatchViewer(grid_shape, patch_shape, pad=pad, is_color = is_color)
    for i in xrange(mat.shape[0]):
        if activation is not None:
            if hasattr(activation[0], '__iter__'):
                act = [a[i] for a in activation]
            else:
                act = activation[i]
        else:
            act = None

        patch = topo_view[i, :]

        rval.add_patch(patch, rescale=rescale,
                       activation=act)
    return rval


class PatchViewer(object):
    def __init__(self, grid_shape, patch_shape, is_color=False, pad = None,
            background = None ):
        if background is None:
            if is_color:
                background = np.zeros((3,))
            else:
                background = 0.
        self.background = background
        assert len(grid_shape) == 2
        assert len(patch_shape) == 2
        for shape in [grid_shape, patch_shape]:
            for elem in shape:
                if not isinstance(elem, (int, long)):
                    raise ValueError("Expected grid_shape and patch_shape to be pairs of ints, but they are %s and %s respectively." % (str(grid_shape), str(patch_shape)))
        self.is_color = is_color
        if pad is None:
            self.pad = (5, 5)
        else:
            self.pad = pad
        #these are the colors of the activation shells
        self.colors = [np.asarray([1, 1, 0]), np.asarray([1, 0, 1]), np.asarray([0,1,0])]

        height = (self.pad[0] * (1 + grid_shape[0]) + grid_shape[0] *
                  patch_shape[0])
        width = (self.pad[1] * (1 + grid_shape[1]) + grid_shape[1] *
                 patch_shape[1])
        self.patch_shape = patch_shape
        self.grid_shape = grid_shape

        image_shape = (height, width, 3)

        self.image = np.zeros(image_shape)
        assert self.image.shape[1] == (self.pad[1] * (1 + self.grid_shape[1]) + self.grid_shape[1] *
                                 self.patch_shape[1])
        self.cur_pos = (0, 0)
        #needed to render in the background color
        self.clear()


    def clear(self):
        if self.is_color:
            for i in xrange(3):
                self.image[:,:,i] = self.background[i] * .5 + .5
        else:
            self.image[:] = self.background * .5 + .5
        self.cur_pos = (0, 0)

    #0 is perfect gray. If not rescale, assumes images are in [-1,1]
    def add_patch(self, patch, rescale=True, recenter=True, activation=None, warn_blank_patch = True):
        """
        :param recenter: if patch has smaller dimensions than self.patch, recenter will pad the
        image to the appropriate size before displaying.
        """
        if warn_blank_patch and (patch.min() == patch.max()) and (rescale or patch.min() == 0.0):
            warnings.warn("displaying totally blank patch")


        if self.is_color:
            assert patch.ndim == 3
            assert patch.shape[-1] == 3
        else:
            assert patch.ndim in [2,3]
            if patch.ndim == 3:
                if patch.shape[-1] != 1:
                    raise ValueError("Expected 2D patch or 3D patch with 1 channel, but got patch with shape "+str(patch.shape))

        if recenter:
            assert patch.shape[0] <= self.patch_shape[0]
            if patch.shape[1] > self.patch_shape[1]:
                raise ValueError("Given patch of width %d but only patches up to width %d fit" \
                        % (patch.shape[1],self.patch_shape[1]))
            rs_pad = (self.patch_shape[0] - patch.shape[0]) / 2
            re_pad = self.patch_shape[0] - rs_pad - patch.shape[0]
            cs_pad = (self.patch_shape[1] - patch.shape[1]) / 2
            ce_pad = self.patch_shape[1] - cs_pad - patch.shape[1]
        else:
            if patch.shape[0:2] != self.patch_shape:
                raise ValueError('Expected patch with shape %s, got %s' %
                                 (str(self.patch_shape), str(patch.shape)))
            rs_pad = 0
            re_pad = 0
            cs_pad = 0
            ce_pad = 0

        temp = patch.copy()

        assert (not np.any(np.isnan(temp))) and (not np.any(np.isinf(temp)))

        if rescale:
            scale = np.abs(temp).max()
            if scale > 0:
                temp /= scale
        else:
            if temp.min() < -1.0 or temp.max() > 1.0:
                raise ValueError('When rescale is set to False, pixel values '
                                 'must lie in [-1,1]. Got [%f, %f].'
                                 % (temp.min(), temp.max()))
        temp *= 0.5
        temp += 0.5

        assert temp.min() >= 0.0
        assert temp.max() <= 1.0

        if self.cur_pos == (0, 0):
            self.clear()

        rs = self.pad[0] + (self.cur_pos[0] *
                            (self.patch_shape[0] + self.pad[0]))
        re = rs + self.patch_shape[0]

        assert self.cur_pos[1] <= self.grid_shape[1]
        cs = self.pad[1] + (self.cur_pos[1] *
                            (self.patch_shape[1] + self.pad[1]))
        ce = cs + self.patch_shape[1]

        assert ce <= self.image.shape[1]

        temp *= (temp > 0)

        if len(temp.shape) == 2:
            temp = temp[:, :, np.newaxis]

        assert ce-ce_pad <= self.image.shape[1]
        self.image[rs + rs_pad:re - re_pad, cs + cs_pad:ce - ce_pad, :] = temp

        if activation is not None:
            if (not isinstance(activation, tuple) and
               not isinstance(activation, list)):
                activation = (activation,)

            for shell, amt in enumerate(activation):
                assert 2 * shell + 2 < self.pad[0]
                assert 2 * shell + 2 < self.pad[1]
                if amt >= 0:
                    act = amt * np.asarray(self.colors[shell])
                    self.image[rs + rs_pad - shell - 1,
                               cs + cs_pad - shell - 1:
                               ce - ce_pad + 1 + shell,
                               :] = act
                    self.image[re - re_pad + shell,
                               cs + cs_pad - 1 - shell:
                               ce - ce_pad + 1 + shell,
                               :] = act
                    self.image[rs + rs_pad - 1 - shell:
                               re - re_pad + 1 + shell,
                               cs + cs_pad - 1 - shell,
                               :] = act
                    self.image[rs + rs_pad - shell - 1:
                               re - re_pad + shell + 1,
                               ce - ce_pad + shell,
                               :] = act

        self.cur_pos = (self.cur_pos[0], self.cur_pos[1] + 1)
        if self.cur_pos[1] == self.grid_shape[1]:
            self.cur_pos = (self.cur_pos[0] + 1, 0)
            if self.cur_pos[0] == self.grid_shape[0]:
                self.cur_pos = (0, 0)

    def addVid(self, vid, rescale=False, subtract_mean=False, recenter=False):
        myvid = vid.copy()
        if subtract_mean:
            myvid -= vid.mean()
        if rescale:
            scale = np.abs(myvid).max()
            if scale == 0:
                scale = 1
            myvid /= scale
        for i in xrange(vid.shape[2]):
            self.add_patch(myvid[:, :, i], rescale=False, recenter=recenter)

    def show(self):
        #image.imview_async(self.image)
        show(self.image)

    def get_img(self):
        #print 'image range '+str((self.image.min(), self.image.max()))
        x = np.cast['uint8'](self.image * 255.0)
        if x.shape[2] == 1:
            x = x[:, :, 0]
        img = Image.fromarray(x)
        return img

    def save(self, path):
        self.get_img().save(path)

    def pick_shape(n, exact = False):
        """
        Returns a shape that fits n elements.
        If exact, fits exactly n elements
        """

        if not isinstance(n,int):
            raise TypeError("n must be an integer, but is "+str(type(n)))

        if exact:

            best_r = -1
            best_c = -1
            best_ratio = 0

            for r in xrange(1,int(np.sqrt(n))+1):
                if n % r != 0:
                    continue
                c = n / r

                ratio = min( float(r)/float(c), float(c)/float(r) )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_r = r
                    best_c = c

            return (best_r, best_c)

        sqrt = np.sqrt(n)
        r = c = int(np.floor(sqrt))
        while r * c < n:
            c += 1
        return (r, c)
    pick_shape = staticmethod(pick_shape)

