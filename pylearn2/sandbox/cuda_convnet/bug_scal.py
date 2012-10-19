from theano import scalar
from theano import function

s0 = scalar.Scalar(dtype = 'float32')(name = 's0')
s1 = scalar.Scalar(dtype = 'float32')(name = 's1')

mx0 = scalar.maximum(s0, s1);  mx0.name = 'mx0'
mx1 = scalar.maximum(mx0, s0); mx1.name = 'mx1'

E = scalar.second(mx1,1);      E.name = 'E'
D = scalar.eq(mx1, mx0);       D.name = 'D'
C = D * E;                     C.name = 'C'
B = scalar.eq(mx0, s1);        B.name = 'B'
A = B * C;                     A.name = 'A'

function([s0, s1], [A, D])
