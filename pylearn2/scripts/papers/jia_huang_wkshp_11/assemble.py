import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

#check that the right files are present
names = os.listdir('.')

if 'features.npy' in names:
    logger.error("Not doing anything, features.npy already exists.")
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
            logger.error(
                """I'm not sure what to do with %s The existence of this file
                makes me think extract_features.yaml has changed I don't want
                to do something incorrect so I'm going to give up.
                """, name)
            quit(-1)



#Do the conversion
logger.info('loading %s', chunk_names[0])
first_chunk = np.load(chunk_names[0])

final_shape = list(first_chunk.shape)

final_shape[0] = 50000

logger.info('making output')
X = np.zeros(final_shape,dtype='float32')

idx = first_chunk.shape[0]

X[0:idx,:] = first_chunk

for i in xrange(1, len(chunk_names)):

    arg = chunk_names[i]
    logger.info('loading %s', arg)

    chunk = np.load(arg)

    chunk_span = chunk.shape[0]

    X[idx:idx+chunk_span,...] = chunk

    idx += chunk_span

logger.info("Saving features.npy...")
np.save('features.npy',X)

logger.info("Deleting the chunks...")
for chunk_name in chunk_names:
    os.remove(chunk_name)
