import numpy as np
import os

#check that the right files are present
names = os.listdir('.')

if 'features.npy' in names:
    print "Not doing anything, features.npy already exists."
    quit(0)

chunk_names = [ 'features_A.npy',
                'features_B.npy',
                'features_C.npy',
                'features_D.npy',
                'features_E.npy' ]

for name in chunk_names:
    assert name in names

for name in chunk_names:
    if name.startswith('features') and name.endswith('.npy'):
        if name not in chunk_names:
            print "I'm not sure what to do with "+name
            print "The existence of this file makes me think extract_features.yaml has changed"
            print "I don't want to do something incorrect so I'm going to give up."
            quit(-1)



#Do the conversion
print 'loading '+chunk_names[0]
first_chunk = np.load(chunk_names[0])

final_shape = list(first_chunk.shape)

final_shape[0] = 50000

print 'making output'
X = np.zeros(final_shape,dtype='float32')

idx = first_chunk.shape[0]

X[0:idx,:] = first_chunk

for i in xrange(1, len(chunk_names)):

    arg = chunk_names[i]
    print 'loading '+arg

    chunk = np.load(arg)

    chunk_span = chunk.shape[0]

    X[idx:idx+chunk_span,...] = chunk

    idx += chunk_span

print "Saving features.npy..."
np.save('features.npy',X)

print "Deleting the chunks..."
for chunk_name in chunk_names:
    os.remove(chunk_name)

