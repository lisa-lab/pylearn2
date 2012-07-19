import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels

class MNIST(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False, shuffle = False,
            one_hot = False, binarize = False):

        if which_set not in ['train','test']:
            if which_set == 'valid':
                raise ValueError("There is no such thing as the MNIST "
"validation set. MNIST consists of 60,000 train examples and 10,000 test"
" examples. If you wish to use a validation set you should divide the train "
"set yourself. The pylearn2 dataset implements and will only ever implement "
"the standard train / test split used in the literature.")
            raise ValueError('Unrecognized which_set value "%s".' %
                    (which_set,)+'". Valid values are ["train","test"].')


        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/mnist/"
            if which_set == 'train':
                im_path = path + 'train-images-idx3-ubyte'
                label_path = path + 'train-labels-idx1-ubyte'
            else:
                assert which_set == 'test'
                im_path = path + 't10k-images-idx3-ubyte'
                label_path = path + 't10k-labels-idx1-ubyte'

            topo_view = read_mnist_images(im_path, dtype='float32')
            y = read_mnist_labels(label_path)

            if binarize:
                topo_view = ( topo_view > 0.5).astype('float32')

            self.one_hot = one_hot
            if one_hot:
                one_hot = N.zeros((y.shape[0],10),dtype='float32')
                for i in xrange(y.shape[0]):
                    one_hot[i,y[i]] = 1.
                y = one_hot

            m, r, c = topo_view.shape
            assert r == 28
            assert c == 28
            topo_view = topo_view.reshape(m,r,c,1)

            if which_set == 'train':
                assert m == 60000
            elif which_set == 'test':
                assert m == 10000
            else:
                assert False


            if center:
                topo_view -= topo_view.mean(axis=0)

            if shuffle:
                self.shuffle_rng = np.random.RandomState([1,2,3])
                for i in xrange(topo_view.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    tmp = topo_view[i,:,:,:]
                    topo_view[i,:,:,:] = topo_view[j,:,:,:]
                    topo_view[j,:,:,:] = tmp
                    tmp = y[i]
                    y[i] = y[j]
                    y[j] = tmp

            view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

            super(MNIST,self).__init__(topo_view = topo_view , y = y)

            assert not N.any(N.isnan(self.X))
        else:
            #data loading is disabled, just make something that defines the right topology
            topo = np.zeros((1,28,28,1))
            super(MNIST,self).__init__(topo_view = topo)
            self.X = None

    def adjust_for_viewer(self, X):
        return N.clip(X*2.-1.,-1.,1.)


class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, center = False, one_hot = False):
        path = "${PYLEARN2_DATA_PATH}/mnist/mnist_rotation_back_image/"+which_set

        obj = serial.load(path)
        X = obj['data']
        X = N.cast['float32'](X)
        y = N.asarray(obj['labels'])

        self.one_hot = one_hot
        if one_hot:
            one_hot = N.zeros((y.shape[0],10),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot

        if center:
            X -= X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))

