from pylearn.datasets import utlc

class Avicenna:
    def __init__(self, tag, typename, which_set):
        assert tag == 'dataset'
        assert typename == 'avicenna'

        train, valid, test = utlc.load_ndarray_dataset('rita')

        if which_set == 'train':
            self.X = train
        elif which_set == 'valid':
            self.X = valid
        elif which_set == 'test':
            self.X = test
        else:
            assert False
