import numpy as N
from PIL import Image

class  PatchViewer:

    def __init__(self,grid_shape, patch_shape, is_color = False):
        assert len(grid_shape) == 2
        assert len(patch_shape) == 2

        self.is_color = is_color

        self.pad = (3,3)

        height  = self.pad[0]*(1+grid_shape[0])+grid_shape[0]*patch_shape[0]
        width = self.pad[1]*(1+grid_shape[1])+grid_shape[1]*patch_shape[1]


        if is_color:
            image_shape = (height, width, 3)
        else:
            image_shape = (height, width)
        #
        self.image = N.zeros( image_shape ) + 0.5
        self.curPos = (0,0)

        self.patch_shape = patch_shape
        self.grid_shape = grid_shape

        #print "Made patch viewer with "+str(grid_shape)+" panels and patch size "+str(patch_shape)

    def clear(self):
		self.image[:] = 0.5
		self.curPos = (0,0)

	#0 is perfect gray. If not rescale, assumes images are in [-1,1]
    def add_patch(self, patch , rescale = True, recenter = False):
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

        self.image[rs+rs_pad:re-re_pad,cs+cs_pad:ce-ce_pad] = temp

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
		self.get_img().show()

    def get_img(self):
        #print 'image range '+str((self.image.min(), self.image.max()))
        return Image.fromarray(N.cast['int8'](self.image*255.0))

    def save(self, path):
        self.get_img().save(path)

    def pickSize(n):
		r = c = int(N.floor(N.sqrt(n)))
		while r * c < n:
			c += 1
		return (r,c)
    pickSize = staticmethod(pickSize)

