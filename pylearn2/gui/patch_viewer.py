import numpy as N
from PIL import Image
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils.image import show

def make_viewer(mat, grid_shape=None, patch_shape=None, activation=None, pad=None, is_color = False, rescale = True):
    """ Given filters in rows, guesses dimensions of patchse
        and nice dimensions for the PatchViewer and returns a PatchViewer
        containing visualizations of the filters"""

    num_channels = 1
    if is_color:
        num_channels = 3

    if grid_shape is None:
        grid_shape = PatchViewer.pick_shape(mat.shape[0] )
    if patch_shape is None:
        assert mat.shape[1] % num_channels == 0
        patch_shape = PatchViewer.pick_shape(mat.shape[1] / num_channels, exact = True)
        assert patch_shape[0] * patch_shape[1] * num_channels == mat.shape[1]
    rval = PatchViewer(grid_shape, patch_shape, pad=pad)
    topo_shape = (patch_shape[0], patch_shape[1], num_channels)
    view_converter = DefaultViewConverter(topo_shape)
    topo_view = view_converter.design_mat_to_topo_view(mat)
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
    def __init__(self, grid_shape, patch_shape, is_color=False, pad = None):
        assert len(grid_shape) == 2
        assert len(patch_shape) == 2
        self.is_color = is_color
        if pad is None:
            self.pad = (5, 5)
        else:
            self.pad = pad
        self.colors = [N.asarray([1, 1, 0]), N.asarray([1, 0, 1]), N.asarray([0,1,0])]

        height = (self.pad[0] * (1 + grid_shape[0]) + grid_shape[0] *
                  patch_shape[0])
        width = (self.pad[1] * (1 + grid_shape[1]) + grid_shape[1] *
                 patch_shape[1])

        image_shape = (height, width, 3)

        self.image = N.zeros(image_shape) + 0.5
        self.cur_pos = (0, 0)

        self.patch_shape = patch_shape
        self.grid_shape = grid_shape

        #print "Made patch viewer with "+str(grid_shape)+" panels and patch
        # size "+str(patch_shape)

    def clear(self):
        self.image[:] = 0.5
        self.cur_pos = (0, 0)

    #0 is perfect gray. If not rescale, assumes images are in [-1,1]
    def add_patch(self, patch, rescale=True, recenter=True, activation=None):
        """
        :param recenter: if patch has smaller dimensions than self.patch, recenter will pad the
        image to the appropriate size before displaying.
        """
        if (patch.min() == patch.max()) and (rescale or patch.min() == 0.0):
            print "Warning: displaying totally blank patch"


        if recenter:
            assert patch.shape[0] <= self.patch_shape[0]
            assert patch.shape[1] <= self.patch_shape[1]
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

        assert (not N.any(N.isnan(temp))) and (not N.any(N.isinf(temp)))

        if rescale:
            scale = N.abs(temp).max()
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
            self.image[:] = 0.5

        rs = self.pad[0] + (self.cur_pos[0] *
                            (self.patch_shape[0] + self.pad[0]))
        re = rs + self.patch_shape[0]

        cs = self.pad[1] + (self.cur_pos[1] *
                            (self.patch_shape[1] + self.pad[1]))
        ce = cs + self.patch_shape[1]

        #print self.cur_pos
        #print cs

        #print (temp.min(), temp.max(), temp.argmax())

        temp *= (temp > 0)

        if len(temp.shape) == 2:
            temp = temp[:, :, N.newaxis]

        self.image[rs + rs_pad:re - re_pad, cs + cs_pad:ce - ce_pad, :] = temp

        if activation is not None:
            if (not isinstance(activation, tuple) and
               not isinstance(activation, list)):
                activation = (activation,)

            for shell, amt in enumerate(activation):
                assert 2 * shell + 2 < self.pad[0]
                assert 2 * shell + 2 < self.pad[1]
                if amt >= 0:
                    act = amt * N.asarray(self.colors[shell])
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
            scale = N.abs(myvid).max()
            if scale == 0:
                scale = 1
            myvid /= scale
        for i in xrange(vid.shape[2]):
            self.add_patch(myvid[:, :, i], rescale=False, recenter=recenter)

    def show(self):
        show(self.image)

    def get_img(self):
        #print 'image range '+str((self.image.min(), self.image.max()))
        x = N.cast['uint8'](self.image * 255.0)
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

        if exact:

            best_r = -1
            best_c = -1
            best_ratio = 0

            for r in xrange(1,int(N.sqrt(n))+1):
                if n % r != 0:
                    continue
                c = n / r

                ratio = min( float(r)/float(c), float(c)/float(r) )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_r = r
                    best_c = c

            return (best_r, best_c)


        r = c = int(N.floor(N.sqrt(n)))
        while r * c < n:
            c += 1
        return (r, c)
    pick_shape = staticmethod(pick_shape)

