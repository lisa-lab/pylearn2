from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np

def show(image, title = "pylearn2 image viewer"):
    def process_job():
        f = plt.figure()
        f.canvas.set_window_title(title)
        im = image
        if len(image.shape) != 3:
            assert len(image.shape) == 2
            im = np.lib.stride_tricks.as_strided(image, image.shape + (3,), image.strides + (0,))
        plt.imshow(im[::-1,::-1,...])
        plt.show()
    p = Process(None, process_job)
    p.start()

