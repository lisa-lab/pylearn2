"""
main idea of this file:

When running mutiple jobs using the same dataset, each job would have to read
the same dataset from ${PYLEARN2_DATA_PATH}, which may cause a great burden on the network.

The solution is to copy the required dataset into ${PYLEARN2_LOCAL_DATA_ROOT} that is
local to the node that is executing the job. The job can thus read locally the dataset
if possible.

Then whenever a new job is run on a node, it searches the required dataset in
${PYLEARN2_LOCAL_DATA_ROOT}. If the dataset is there, use it directly, otherwise, copy
it from the remote server.

workflow:
1. Check if ${PYLEARN2_LOCAL_DATA_ROOT} exists, if not, create it.
   This step needs a lock under the dir '/tmp/'
2. Check if the required dataset D exists under ${PYLEARN2_LOCAL_DATA_ROOT}.
   If yes, load it.
   Otherwise:
       Check if the local node has enough space to maintain a local copy of D.
          If no, read D directly from ${PYLEARN2_DATA_PATH}. END
          If yes:
            Get a lock under ${PYLEARN2_LOCAL_DATA_ROOT}
            Check if D has been copied there by some other jobs.
               If yes, load it, release the lock. END
               if no,
                  Check if the local node has enough space to maintain a local copy of D.
                  If no, read D directly from ${PYLEARN2_DATA_PATH}. release lock. END
                  If yes, copy D from remote server, read it locally. release lock. END

Whenever a folder or a dataset copy is created locally, it is granted the same access as it
has under ${PYLEARN2_LOCAL_DATA_ROOT}. This is gauranteed by default copy.
"""
import os
from pylearn2.utils import string_utils
import theano.gof.compilelock as compilelock
import shutil

class DatasetLoader:
    def __init__(self):
        
        # dataset paths
        # 
        self.dataset_local_dir = string_utils.preprocess('${PYLEARN2_LOCAL_DATA_ROOT}')
        
        # only allow single remote dir
        self.dataset_remote_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')

        self.tmp_dir = '/tmp'
        
        # lock to create local copies of datasets inside current_local_dir
        self.dir_lock_in = os.path.join(self.dataset_local_dir, 'lock')
        
        # lock used to create current_local_dir
        self.dir_lock_out = os.path.join(self.tmp_dir, 'lock')
        
        self.check_local_dataset_repo()
        
    def check_local_dataset_repo(self):
        
        if not os.path.exists(self.dataset_local_dir): 
            self.get_lock(self.dir_lock_out)
            if not os.path.exists(self.dataset_local_dir):
                print 'created local repo ', self.dataset_local_dir 
                os.makedirs(self.dataset_local_dir)
            self.release_lock()
        
    def get_lock(self, dir_lock):
        compilelock.get_lock(dir_lock)
        
    def release_lock(self):
        compilelock.release_lock()
        
    def check_existence(self, fname):
        """
        check if the dataset is already there.
        if yes, return True, otherwise return False
        """
        return os.path.exists(fname) 
    
    def load_dataset(self, fname):
        # called by e.g. cifar10.py
        
        local_fname = os.path.join(self.dataset_local_dir,
                                   os.path.relpath(fname, self.dataset_remote_dir))
        remote_fname = fname
        
        existing = self.check_existence(local_fname)
                                                
        if existing:
            dict = self.load_existing_dataset(local_fname)
        else:
            dict = self.load_missing_dataset(remote_fname, local_fname)
        return dict

    def force_load_from_server(self, fname):
        dataset_file = open(fname, 'rb')
        dict = cPickle.load(dataset_file)
        dataset_file.close()
        return dict
        
    def load_existing_dataset(self, fname):
        print 'loading datasets from local dataset repo...'
        fo = open(fname, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def copy_from_server_to_local(self, remote_fname, local_fname):
        print "copying the required dataset from remote repo..."
        head, tail = os.path.split(local_fname)
        head += '/'
        if not os.path.exists(head):
            os.makedirs(os.path.dirname(head))

        command = 'cp ' + remote_fname + ' ' + local_fname
        os.system(command)
        
        
    def load_missing_dataset(self, remote_fname, local_fname):
        has_enough_space = self.check_enough_space(remote_fname, local_fname)
        
        if has_enough_space:
            self.get_lock(self.dir_lock_in)
            has_enough_space = self.check_enough_space(remote_fname,
                                                       local_fname)
            existed = self.check_existence(local_fname)
            if has_enough_space and not existed:
                self.copy_from_server_to_local(remote_fname, local_fname)
                self.release_lock()
                dict = self.load_existing_dataset(local_fname)
                
            elif existed:
                self.release_lock()
                dict = self.load_existing_dataset(local_fname)
            
            else:
                print "not enough space in local machine, load directly from data server"
                dict = self.force_load_from_server(fname)
        else:
            print "not enough space in local machine, load directly from data server"
            dict = self.force_load_from_server(fname)
            
        return dict
    
    def get_storage_requirement(self, fname):
        """
        from remote server
        """
        rval = os.path.getsize(fname)
        return rval

    def disk_usage(self, path):
        """
        Return disk usage statistics about the given path.
        
        Returned valus is a named tuple with attributes 'total', 'used' and
        'free', which are the amount of total, used and free space, in bytes.
        """
        
        st = os.statvfs(path)
        free = st.f_bavail * st.f_frsize
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        
        return free
        
    def check_enough_space(self, remote_fname, local_fname):
        """
        check if the local disk have enough space to store the dataset
        """
        storage_need = self.get_storage_requirement(remote_fname)
        storage_remain = self.disk_usage(self.tmp_dir)
        
        return storage_remain > storage_need