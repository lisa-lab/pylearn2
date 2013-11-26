#from pylearn2.datasets.csv_dataset import CSVDataset
from csv_dataset import CSVDataset
import numpy as np

def test_loading():
    d = CSVDataset(path = 'test.csv', expect_headers = False)
    assert(np.array_equal( d.X, np.array([[1., 2., 3.], [4., 5., 6.]])))
    assert(np.array_equal( d.y, np.array([0., 1.])))
    
    

