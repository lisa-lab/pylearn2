import theano.tensor as T
from theano import function

s0 = T.scalar('s0')
s1 = T.scalar('s1')

mx0 = T.maximum(s0, s1);  mx0.name = 'mx0'
mx1 = T.maximum(mx0, s0); mx1.name = 'mx1'

E = T.ones_like(mx1);     E.name = 'E'
D = T.eq(mx1, mx0);       D.name = 'D'
C = D * E;                C.name = 'C'
B = T.eq(mx0, s1);        B.name = 'B'
A = B * C;                A.name = 'A'

function([s0, s1], [A, D])
