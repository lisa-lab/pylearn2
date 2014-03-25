import numpy as np
import pylearn2.datasets.hepatitis as hepatitis
from pylearn2.testing.skip import skip_if_no_data


def test_hepatitis():
    """test hepatitis dataset"""
    skip_if_no_data()
    data = hepatitis.Hepatitis()
    assert data.X is not None
    assert np.all(data.X != np.inf)
    assert np.all(data.X != np.nan)

def test_data():
    """test data in hepatitis.py against backup here"""
    assert hepatitis_data == hepatitis.hepatitis_data

hepatitis_data = \
"""2,30,2,1,2,2,2,2,1,2,2,2,2,2,1.00,85,18,4.0,?,1
2,50,1,1,2,1,2,2,1,2,2,2,2,2,0.90,135,42,3.5,?,1
2,78,1,2,2,1,2,2,2,2,2,2,2,2,0.70,96,32,4.0,?,1
2,31,1,?,1,2,2,2,2,2,2,2,2,2,0.70,46,52,4.0,80,1
2,34,1,2,2,2,2,2,2,2,2,2,2,2,1.00,?,200,4.0,?,1
2,34,1,2,2,2,2,2,2,2,2,2,2,2,0.90,95,28,4.0,75,1
1,51,1,1,2,1,2,1,2,2,1,1,2,2,?,?,?,?,?,1
2,23,1,2,2,2,2,2,2,2,2,2,2,2,1.00,?,?,?,?,1
2,39,1,2,2,1,2,2,2,1,2,2,2,2,0.70,?,48,4.4,?,1
2,30,1,2,2,2,2,2,2,2,2,2,2,2,1.00,?,120,3.9,?,1
2,39,1,1,1,2,2,2,1,1,2,2,2,2,1.30,78,30,4.4,85,1
2,32,1,2,1,1,2,2,2,1,2,1,2,2,1.00,59,249,3.7,54,1
2,41,1,2,1,1,2,2,2,1,2,2,2,2,0.90,81,60,3.9,52,1
2,30,1,2,2,1,2,2,2,1,2,2,2,2,2.20,57,144,4.9,78,1
2,47,1,1,1,2,2,2,2,2,2,2,2,2,?,?,60,?,?,1
2,38,1,1,2,1,1,1,2,2,2,2,1,2,2.00,72,89,2.9,46,1
2,66,1,2,2,1,2,2,2,2,2,2,2,2,1.20,102,53,4.3,?,1
2,40,1,1,2,1,2,2,2,1,2,2,2,2,0.60,62,166,4.0,63,1
2,38,1,2,2,2,2,2,2,2,2,2,2,2,0.70,53,42,4.1,85,2
2,38,1,1,1,2,2,2,1,1,2,2,2,2,0.70,70,28,4.2,62,1
2,22,2,2,1,1,2,2,2,2,2,2,2,2,0.90,48,20,4.2,64,1
2,27,1,2,2,1,1,1,1,1,1,1,2,2,1.20,133,98,4.1,39,1
2,31,1,2,2,2,2,2,2,2,2,2,2,2,1.00,85,20,4.0,100,1
2,42,1,2,2,2,2,2,2,2,2,2,2,2,0.90,60,63,4.7,47,1
2,25,2,1,1,2,2,2,2,2,2,2,2,2,0.40,45,18,4.3,70,1
2,27,1,1,2,1,1,2,2,2,2,2,2,2,0.80,95,46,3.8,100,1
2,49,1,1,1,1,1,1,2,1,2,1,2,2,0.60,85,48,3.7,?,1
2,58,2,2,2,1,2,2,2,1,2,1,2,2,1.40,175,55,2.7,36,1
2,61,1,1,2,1,2,2,1,1,2,2,2,2,1.30,78,25,3.8,100,1
2,51,1,1,1,1,1,2,2,2,2,2,2,2,1.00,78,58,4.6,52,1
1,39,1,1,1,1,1,2,2,1,2,2,2,2,2.30,280,98,3.8,40,1
1,62,1,1,2,1,1,2,?,?,2,2,2,2,1.00,?,60,?,?,1
2,41,2,2,1,1,1,1,2,2,2,2,2,2,0.70,81,53,5.0,74,1
2,26,2,1,2,2,2,2,2,1,2,2,2,2,0.50,135,29,3.8,60,1
2,35,1,2,2,1,2,2,2,2,2,2,2,2,0.90,58,92,4.3,73,1
1,37,1,2,2,1,2,2,2,2,2,1,2,2,0.60,67,28,4.2,?,1
2,23,1,2,2,1,1,1,2,2,1,2,2,2,1.30,194,150,4.1,90,1
2,20,2,1,2,1,1,1,1,1,1,1,2,2,2.30,150,68,3.9,?,1
2,42,1,1,2,2,2,2,2,2,2,2,2,2,1.00,85,14,4.0,100,1
2,65,1,2,2,1,1,2,2,1,1,1,1,2,0.30,180,53,2.9,74,2
2,52,1,1,1,2,2,2,2,2,2,2,2,2,0.70,75,55,4.0,21,1
2,23,1,2,2,2,2,2,?,?,?,?,?,?,4.60,56,16,4.6,?,1
2,33,1,2,2,2,2,2,2,2,2,2,2,2,1.00,46,90,4.4,60,1
2,56,1,1,2,1,2,2,2,2,2,2,2,2,0.70,71,18,4.4,100,1
2,34,1,2,2,2,2,2,2,2,2,2,2,2,?,?,86,?,?,1
2,28,1,2,2,1,1,2,2,2,2,2,2,2,0.70,74,110,4.4,?,1
2,37,1,1,2,2,2,2,2,1,2,1,2,2,0.60,80,80,3.8,?,1
2,28,2,2,2,1,1,2,2,1,2,2,2,2,1.80,191,420,3.3,46,1
2,36,1,1,2,2,2,2,2,2,1,2,2,2,0.80,85,44,4.2,85,1
2,38,1,2,1,1,1,1,2,2,2,1,2,2,0.70,125,65,4.2,77,1
2,39,1,1,2,2,2,2,2,2,2,2,2,2,0.90,85,60,4.0,?,1
2,39,1,2,2,2,2,2,2,2,2,2,2,2,1.00,85,20,4.0,?,1
2,44,1,2,2,2,2,2,2,2,2,2,2,2,0.60,110,145,4.4,70,1
2,40,1,2,1,1,2,2,2,1,1,2,2,2,1.20,85,31,4.0,100,1
2,30,1,2,2,1,2,2,2,2,2,2,2,2,0.70,50,78,4.2,74,1
2,37,1,1,2,1,1,1,2,2,2,2,2,2,0.80,92,59,?,?,1
2,34,1,1,2,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1
2,30,1,2,1,2,2,2,2,2,2,2,2,2,0.70,52,38,3.9,52,1
2,64,1,2,1,1,1,2,1,1,2,2,2,2,1.00,80,38,4.3,74,1
2,45,2,1,2,1,1,2,2,2,1,2,2,2,1.00,85,75,?,?,1
2,37,1,2,2,2,2,2,2,2,2,2,2,2,0.70,26,58,4.5,100,1
2,32,1,2,2,2,2,2,2,2,2,2,2,2,0.70,102,64,4.0,90,1
2,32,1,2,2,1,1,1,2,2,2,1,2,1,3.50,215,54,3.4,29,1
2,36,1,1,2,2,2,2,1,1,1,2,2,2,0.70,164,44,3.1,41,1
2,49,1,2,2,1,1,2,2,2,2,2,2,2,0.80,103,43,3.5,66,1
2,27,1,2,2,2,2,2,2,2,2,2,2,2,0.80,?,38,4.2,?,1
2,56,1,1,2,2,2,2,2,2,2,2,2,2,0.70,62,33,3.0,?,1
1,57,1,2,2,1,1,1,2,2,2,1,1,2,4.10,?,48,2.6,73,1
2,39,1,2,2,1,2,2,2,2,2,2,2,2,1.00,34,15,4.0,54,1
2,44,1,1,2,1,1,2,2,2,2,2,2,2,1.60,68,68,3.7,?,1
2,24,1,2,2,2,2,2,2,2,2,2,2,2,0.80,82,39,4.3,?,1
1,34,1,1,2,1,1,2,1,1,2,1,2,2,2.80,127,182,?,?,1
2,51,1,2,2,1,1,1,?,?,?,?,?,?,0.90,76,271,4.4,?,1
2,36,1,1,2,1,1,1,2,1,2,2,2,2,1.00,?,45,4.0,57,1
2,50,1,2,2,2,2,2,2,2,2,2,2,2,1.50,100,100,5.3,?,1
2,32,1,1,1,1,1,2,2,2,2,2,2,2,1.00,55,45,4.1,56,1
1,58,1,2,2,1,2,2,1,1,1,1,2,2,2.00,167,242,3.3,?,1
2,34,2,1,1,2,2,2,2,1,2,2,2,2,0.60,30,24,4.0,76,1
2,34,1,1,2,1,2,2,1,1,2,1,2,2,1.00,72,46,4.4,57,1
2,28,1,2,2,2,2,2,2,2,2,2,2,2,0.70,85,31,4.9,?,1
2,23,1,2,2,1,1,1,2,2,2,2,2,2,0.80,?,14,4.8,?,1
2,36,1,2,2,2,2,2,2,2,2,2,2,2,0.70,62,224,4.2,100,1
2,30,1,1,2,2,2,2,2,2,2,2,2,2,0.70,100,31,4.0,100,1
2,67,2,1,2,1,1,2,2,2,?,?,?,?,1.50,179,69,2.9,?,1
2,62,2,2,2,1,1,2,2,1,2,1,2,2,1.30,141,156,3.9,58,1
2,28,1,1,2,1,1,1,2,1,2,2,2,2,1.60,44,123,4.0,46,1
1,44,1,1,2,1,1,2,2,2,1,2,2,1,0.90,135,55,?,41,2
1,30,1,2,2,1,1,1,2,1,2,1,1,1,2.50,165,64,2.8,?,2
1,38,1,1,2,1,1,1,2,1,2,1,1,1,1.20,118,16,2.8,?,2
2,38,1,1,2,1,1,1,1,1,2,2,2,2,0.60,76,18,4.4,84,2
2,50,2,1,2,1,2,2,1,1,1,1,2,2,0.90,230,117,3.4,41,2
1,42,1,1,2,1,1,1,2,2,1,1,2,1,4.60,?,55,3.3,?,2
2,33,1,2,2,2,2,2,?,?,2,2,2,2,1.00,?,60,4.0,?,2
2,52,1,1,2,2,2,2,2,2,2,2,2,2,1.50,?,69,2.9,?,2
1,59,1,1,2,1,1,2,2,1,1,1,2,2,1.50,107,157,3.6,38,2
2,40,1,1,1,1,1,1,1,1,2,2,2,2,0.60,40,69,4.2,67,2
2,30,1,1,2,1,1,2,2,1,2,1,2,2,0.80,147,128,3.9,100,2
2,44,1,1,2,1,1,2,1,1,2,1,2,2,3.00,114,65,3.5,?,2
1,47,1,2,2,2,2,2,2,2,2,1,2,1,2.00,84,23,4.2,66,2
2,60,1,1,2,1,2,2,1,1,1,1,2,2,?,?,40,?,?,2
1,48,1,1,2,1,1,2,2,1,2,1,1,1,4.80,123,157,2.7,31,2
2,22,1,2,2,2,2,2,2,2,2,2,2,2,0.70,?,24,?,?,2
2,27,1,1,2,1,2,2,2,1,2,2,2,2,2.40,168,227,3.0,66,2
2,51,1,1,2,1,1,1,2,1,1,1,2,1,4.60,215,269,3.9,51,2
1,47,1,2,2,1,1,2,2,1,2,2,1,1,1.70,86,20,2.1,46,2
2,25,1,2,2,2,2,2,2,2,2,2,2,2,0.60,?,34,6.4,?,2
1,35,1,1,2,1,2,2,?,?,1,1,1,2,1.50,138,58,2.6,?,2
2,45,1,1,2,1,1,1,2,2,2,2,2,2,2.30,?,648,?,?,2
2,54,1,1,1,2,2,2,1,1,2,2,2,2,1.00,155,225,3.6,67,2
1,33,1,1,2,1,1,2,2,2,2,2,1,2,0.70,63,80,3.0,31,2
2,7,1,2,2,2,2,2,2,1,1,2,2,2,0.70,256,25,4.2,?,2
1,42,1,1,1,1,1,2,2,2,2,1,2,2,0.50,62,68,3.8,29,2
2,52,1,1,2,1,2,2,2,2,2,2,2,2,1.00,85,30,4.0,?,2
2,45,1,1,2,1,2,2,2,1,1,2,2,2,1.20,81,65,3.0,?,1
2,36,1,1,2,2,2,2,2,2,2,2,2,2,1.10,141,75,3.3,?,2
2,69,2,2,2,1,2,2,2,2,2,2,2,2,3.20,119,136,?,?,2
2,24,1,1,2,1,2,2,2,2,2,2,2,2,1.00,?,34,4.1,?,2
2,50,1,2,2,2,2,2,2,2,2,2,2,2,1.00,139,81,3.9,62,2
1,61,1,1,2,1,1,2,?,?,2,1,2,2,?,?,?,?,?,2
2,54,1,2,2,1,2,2,1,1,2,2,2,2,3.20,85,28,3.8,?,2
1,56,1,1,2,1,1,1,1,1,2,1,2,2,2.90,90,153,4.0,?,2
2,20,1,1,2,1,1,1,2,2,2,1,1,2,1.00,160,118,2.9,23,2
2,42,1,2,2,2,2,2,2,2,1,2,2,2,1.50,85,40,?,?,2
2,37,1,1,2,1,2,2,2,2,2,1,2,2,0.90,?,231,4.3,?,2
2,50,1,2,2,2,2,2,2,1,1,1,2,2,1.00,85,75,4.0,72,2
2,34,2,2,2,1,1,1,1,1,2,1,2,2,0.70,70,24,4.1,100,2
2,28,1,2,2,1,1,1,?,?,2,1,1,2,1.00,?,20,4.0,?,2
1,50,1,2,2,1,2,2,2,1,1,2,1,1,2.80,155,75,2.4,32,2
2,54,1,1,2,1,1,2,2,2,2,2,1,2,1.20,85,92,3.1,66,2
1,57,1,1,2,1,1,2,2,2,2,1,1,2,4.60,82,55,3.3,30,2
2,54,1,2,2,2,2,2,2,2,2,2,2,2,1.00,85,30,4.5,0,2
1,31,1,1,2,1,1,1,2,2,1,2,2,2,8.00,?,101,2.2,?,2
2,48,1,2,2,1,1,1,2,1,2,1,2,2,2.00,158,278,3.8,?,2
2,72,1,2,1,1,2,2,2,1,2,2,2,2,1.00,115,52,3.4,50,2
1,38,1,1,2,2,2,2,2,1,2,2,2,2,0.40,243,49,3.8,90,2
2,25,1,2,2,1,2,2,1,1,1,1,1,1,1.30,181,181,4.5,57,2
2,51,1,2,2,2,2,2,1,1,2,1,2,2,0.80,?,33,4.5,?,2
2,38,1,2,2,2,2,2,2,1,2,1,2,1,1.60,130,140,3.5,56,2
1,47,1,2,2,1,1,2,2,1,2,1,1,1,1.00,166,30,2.6,31,2
2,45,1,2,1,2,2,2,2,2,2,2,2,2,1.30,85,44,4.2,85,2
2,36,1,1,2,1,1,1,1,1,2,1,2,1,1.70,295,60,2.7,?,2
1,54,1,1,2,1,1,2,?,?,1,2,1,2,3.90,120,28,3.5,43,2
2,51,1,2,2,1,2,2,2,1,1,1,2,1,1.00,?,20,3.0,63,2
1,49,1,1,2,1,1,2,2,2,1,1,2,2,1.40,85,70,3.5,35,2
1,45,1,2,2,1,1,1,2,2,2,1,1,2,1.90,?,114,2.4,?,2
2,31,1,1,2,1,2,2,2,2,2,2,2,2,1.20,75,173,4.2,54,2
1,41,1,2,2,1,2,2,2,1,1,1,2,1,4.20,65,120,3.4,?,2
1,70,1,1,2,1,1,1,?,?,?,?,?,?,1.70,109,528,2.8,35,2
2,20,1,1,2,2,2,2,2,?,2,2,2,2,0.90,89,152,4.0,?,2
2,36,1,2,2,2,2,2,2,2,2,2,2,2,0.60,120,30,4.0,?,2
1,46,1,2,2,1,1,1,2,2,2,1,1,1,7.60,?,242,3.3,50,2
2,44,1,2,2,1,2,2,2,1,2,2,2,2,0.90,126,142,4.3,?,2
2,61,1,1,2,1,1,2,1,1,2,1,2,2,0.80,75,20,4.1,?,2
2,53,2,1,2,1,2,2,2,2,1,1,2,1,1.50,81,19,4.1,48,2
1,43,1,2,2,1,2,2,2,2,1,1,1,2,1.20,100,19,3.1,42,2"""
