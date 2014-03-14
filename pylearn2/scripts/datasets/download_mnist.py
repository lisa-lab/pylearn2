import os
import urllib
import gzip
import logging

logger = logging.getLogger(__name__)

assert 'PYLEARN2_DATA_PATH' in os.environ, "PYLEARN2_DATA_PATH not defined"
mnist_path = os.path.join(os.environ['PYLEARN2_DATA_PATH'], "mnist")

if not os.path.isdir(mnist_path):
    logger.info("creating path: %s" + mnist_path)
    os.makedirs(mnist_path)

in_dir = os.listdir(mnist_path)
mnist_files = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
               "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]
mnist_url = "http://yann.lecun.com/exdb/mnist/"

if not all([f in in_dir for f in mnist_files]) or in_dir == []:
    logger.info("Downloading MNIST data...")
    gz_in = [os.path.join(mnist_path, f + ".gz") for f in mnist_files]
    gz_out = [os.path.join(mnist_path, f)for f in mnist_files]
    mnist_url = ["".join([mnist_url, f, ".gz"]) for f in mnist_files]

    for g_in, g_out, m_url in zip(gz_in, gz_out, mnist_url):
        logger.info("Downloading %s... %s", m_url,
                    urllib.urlretrieve(m_url, filename=g_in))
        logger.info(" Done")

        with gzip.GzipFile(g_in) as f_gz:
            data = f_gz.read()

        with open(g_out, 'wb') as f_out:
            f_out.write(data)

    logger.info("Done downloading MNIST")
else:
    logger.info("MNIST files already in PYLEARN2_DATA_PATH")
