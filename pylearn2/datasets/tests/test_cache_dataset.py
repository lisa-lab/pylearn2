import unittest
from pylearn2.datasets import cifar10 
import multiprocessing
import os

class TestDatasetCache(object):
    """
    search a dataset locally under dir '/tmp/pylean2_local_cache/'
    if the desired dataset exists, load it locally
    if not, load it remotely to the cache first and then load it locally

    whenever a dataset if required, search in the cache first. 
    """
    
    def test_main_multi_thread(self):
        jobs = []
        for i in range(3):
            p = multiprocessing.Process(target=self.worker, args=(i,))
            jobs.append(p)
            p.start()

    def worker(self, num):
        """thread worker function"""
        print 'Worker:', num
        trainset = cifar10.CIFAR10(which_set="train")

    def test_main(self):
        trainset = cifar10.CIFAR10(which_set="train")
        
if __name__ == '__main__':
    instance = TestDatasetCache()
    instance.test_main_multi_thread()
    #instance.test_main()
    #os.system()