import numpy as N
from PIL import Image
import os

def make_viewer(mat, grid_shape  = None, patch_shape = None, activation = None):
    """ Given filters in rows, guesses dimensions of patchse
        and nice dimensions for the PatchViewer and returns a PatchViewer
        containing visualizations of the filters"""
    if grid_shape is None:
        grid_shape = PatchViewer.pickSize(mat.shape[0])

    if patch_shape is None:
        patch_shape = PatchViewer.pickSize(mat.shape[1])


    rval = PatchViewer( grid_shape, patch_shape)

    patch_shape = (patch_shape[0], patch_shape[1], 1)

    for i in xrange(mat.shape[0]):
        #rval.add_patch( N.ones(patch_shape) )
        if activation is not None:
            if isinstance(activation,list) or isinstance(activation,tuple):
                act = [ a[i] for a in activation ]
            else:
                act = activation[i]
        else:
            act = None

        rval.add_patch( mat[i,:].reshape(*patch_shape), rescale = True , activation = act)
    #

    return rval
#

class  PatchViewer(object):

    def __init__(self,grid_shape, patch_shape, is_color = False):
        assert len(grid_shape) == 2
        assert len(patch_shape) == 2

        self.is_color = is_color

        self.pad = (5,5)

        self.colors = [ N.asarray([1,1,0]),N.asarray([1,0,1]) ]

        height  = self.pad[0]*(1+grid_shape[0])+grid_shape[0]*patch_shape[0]
        width = self.pad[1]*(1+grid_shape[1])+grid_shape[1]*patch_shape[1]


        image_shape = (height, width, 3)

        self.image = N.zeros( image_shape ) + 0.5
        self.curPos = (0,0)

        self.patch_shape = patch_shape
        self.grid_shape = grid_shape

        #print "Made patch viewer with "+str(grid_shape)+" panels and patch size "+str(patch_shape)

    def clear(self):
		self.image[:] = 0.5
		self.curPos = (0,0)

	#0 is perfect gray. If not rescale, assumes images are in [-1,1]
    def add_patch(self, patch , rescale = True, recenter = False, activation = None):

        if (patch.min() == patch.max()) and (rescale or patch.min() == 0.0):
            print "Warning: displaying totally blank patch"

        if patch.shape[0:2] != self.patch_shape:
            raise ValueError('Expected patch with shape '+str(self.patch_shape)+', got '+str(patch.shape))

        if recenter:
            assert patch.shape[0] < self.patch_shape[0]
            assert patch.shape[1] < self.patch_shape[1]
            rs_pad = (self.patch_shape[0] - patch.shape[0])/2
            re_pad = self.patch_shape[0] - rs_pad - patch.shape[0]
            cs_pad = (self.patch_shape[1] - patch.shape[1])/2
            ce_pad = self.patch_shape[1] - cs_pad - patch.shape[1]
        else:
            if patch.shape[0:2] != tuple(self.patch_shape):
                raise Exception("Expected patch of shape "+str(self.patch_shape)+", got one of shape "+str(patch.shape))
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
                raise ValueError('When rescale is set to False, pixel values must lie in [-1,1]. Got ['+str(temp.min())+','+str(temp.max())+']')
            #
        #

        temp *= 0.5
        temp += 0.5

        assert temp.min() >= 0.0
        assert temp.max() <= 1.0


        if self.curPos == (0,0):
			self.image[:] = 0.5

        rs = self.pad[0] + self.curPos[0] * (self.patch_shape[0]+self.pad[0])
        re = rs + self.patch_shape[0]

        cs = self.pad[1] + self.curPos[1] * (self.patch_shape[1]+self.pad[1])
        ce = cs + self.patch_shape[1]

		#print self.curPos
		#print cs

		#print (temp.min(), temp.max(), temp.argmax())

        temp *= (temp > 0)

        if len(temp.shape) == 2:
            #is there a clean way to do this generally?
            #this is just meant to make the next line not crash, numpy is too retarded to realize that an mxn array can be assigned to an mxnx1 array
            numpy_sucks = N.zeros((temp.shape[0],temp.shape[1],1),dtype=temp.dtype)
            numpy_sucks[:,:,0] = temp
            temp = numpy_sucks
        #endif

        self.image[rs+rs_pad:re-re_pad,cs+cs_pad:ce-ce_pad,:] = temp

        if activation is not None:
            if not ( isinstance(activation,tuple) or isinstance(activation,list)):
                activation = (activation,)

            for shell, amt in enumerate(activation):
                assert 2*shell+2 < self.pad


                if amt >= 0:

                    act = amt * N.asarray(self.colors[shell])


                    self.image[rs+rs_pad-shell-1,cs+cs_pad-shell-1:ce-ce_pad+1+shell,:] = act
                    self.image[re-re_pad+shell,cs+cs_pad-1-shell:ce-ce_pad+1+shell,:] = act
                    self.image[rs+rs_pad-1-shell:re-re_pad+1+shell,cs+cs_pad-1-shell,:] = act
                    self.image[rs+rs_pad-shell-1:re-re_pad+shell+1,ce-ce_pad+shell,:] = act
                #
            #
        #


        self.curPos = (self.curPos[0], self.curPos[1]+1)
        if self.curPos[1]  == self.grid_shape[1]:
			self.curPos = (self.curPos[0]+1,0)
			if self.curPos[0]  == self.grid_shape[0]:
				self.curPos = (0,0)


    def addVid(self, vid, rescale=False, subtract_mean=False, recenter=False):
        myvid = vid.copy()
        if subtract_mean:
            myvid -= vid.mean()
        if rescale:
            scale = N.abs(myvid).max()
            if scale == 0:
				scale = 1
            myvid /= scale
        for i in xrange(0,vid.shape[2]):
				self.add_patch(myvid[:,:,i],rescale=False, recenter=recenter)


    def show(self):
        try:
            #if os.env['USER'] == 'ia3n':
            #    raise Exception("PIL on Ian's home machine has started using ImageMagick which is a pain to use")
            img = self.get_img()
            img.show()
        except:
            print "Warning, your version of PIL sucks"
            import matplotlib.pyplot
            matplotlib.pyplot.imshow(self.image)
            matplotlib.pyplot.show()
            print 'waiting'
            x = raw_input()
            print 'running'


    def get_img(self):
        #print 'image range '+str((self.image.min(), self.image.max()))
        x = N.cast['int8'](self.image*255.0)
        if x.shape[2] == 1:
            x = x[:,:,0]
        img = Image.fromarray(x)
        return img

    def save(self, path):
        self.get_img().save(path)

    def pickSize(n):
		r = c = int(N.floor(N.sqrt(n)))
		while r * c < n:
			c += 1
		return (r,c)
    pickSize = staticmethod(pickSize)

