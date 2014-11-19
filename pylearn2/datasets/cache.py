"""
Dataset preloading tool

This file provides the ability to make a local cache of a dataset or
part of it. It is meant to help in the case where multiple jobs are
reading the same dataset from ${PYLEARN2_DATA_PATH}, which may cause a
great burden on the network.

With this file, it is possible to make a local copy
(in ${PYLEARN2_LOCAL_DATA_PATH}) of any required file and have multiple
processes use it simultaneously instead of each acquiring its own copy
over the network.

Whenever a folder or a dataset copy is created locally, it is granted
the same access as it has under ${PYLEARN2_LOCAL_DATA_PATH}. This is
gauranteed by default copy.

"""

import atexit
import logging
import os
import stat
import time

import theano.gof.compilelock as compilelock

from pylearn2.utils import string_utils


log = logging.getLogger(__name__)


class LocalDatasetCache:

    """
    A local cache for remote files for faster access and reducing
    network stress.
    """

    def __init__(self):
        default_path = '${PYLEARN2_DATA_PATH}'
        local_path = '${PYLEARN2_LOCAL_DATA_PATH}'
        self.pid = os.getpid()

        try:
            self.dataset_remote_dir = string_utils.preprocess(default_path)
            self.dataset_local_dir = string_utils.preprocess(local_path)
        except (ValueError, string_utils.NoDataPathError,
                string_utils.EnvironmentVariableError):
            # Local cache seems to be deactivated
            self.dataset_remote_dir = ""
            self.dataset_local_dir = ""

        if self.dataset_remote_dir == "" or self.dataset_local_dir == "":
            log.debug("Local dataset cache is deactivated")

    def cache_file(self, filename):
        """
        Caches a file locally if possible. If caching was succesfull, or if
        the file was previously successfully cached, this method returns the
        path to the local copy of the file. If not, it returns the path to
        the original file.

        Parameters
        ----------
        filename : string
            Remote file to cache locally

        Returns
        -------
        output : string
            Updated (if needed) filename to use to access the remote
            file.
        """

        remote_name = string_utils.preprocess(filename)

        # Check if a local directory for data has been defined. Otherwise,
        # do not locally copy the data
        if self.dataset_local_dir == "":
            return filename

        common_msg = ("Message from Pylearn2 local cache of dataset"
                      "(specified by the environment variable "
                      "PYLEARN2_LOCAL_DATA_PATH): ")
        # Make sure the file to cache exists and really is a file
        if not os.path.exists(remote_name):
            log.error("Error : Specified file %s does not exist" %
                      remote_name)
            return filename

        if not os.path.isfile(remote_name):
            log.error("Error : Specified name %s is not a file" %
                      remote_name)
            return filename

        if not remote_name.startswith(self.dataset_remote_dir):
            log.warning(
                common_msg +
                "We cache in the local directory only what is"
                " under $PYLEARN2_DATA_PATH: %s" %
                remote_name)
            return filename

        # Create the $PYLEARN2_LOCAL_DATA_PATH folder if needed
        self.safe_mkdir(self.dataset_local_dir,
                        (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                         stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                         stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH))

        # Determine local path to which the file is to be cached
        local_name = os.path.join(self.dataset_local_dir,
                                  os.path.relpath(remote_name,
                                                  self.dataset_remote_dir))

        # Create the folder structure to receive the remote file
        local_folder = os.path.split(local_name)[0]
        self.safe_mkdir(local_folder)

        # Acquire writelock on the local file to prevent the possibility
        # of any other process modifying it while we cache it if needed.
        # Also, if another process is currently caching the same file,
        # it forces the current process to wait for it to be done before
        # using the file.
        if not os.access(local_folder, os.W_OK):
            log.warning(common_msg +
                        "Local folder %s isn't writable."
                        " This is needed for synchronization."
                        " We will use the remote version."
                        " Manually fix the permission."
                        % local_folder)
            return filename
        self.get_writelock(local_name)

        # If the file does not exist locally, consider creating it
        if not os.path.exists(local_name):

            # Check that there is enough space to cache the file
            if not self.check_enough_space(remote_name, local_name):
                log.warning(common_msg +
                            "File %s not cached: Not enough free space" %
                            remote_name)
                self.release_writelock()
                return filename

            # There is enough space; make a local copy of the file
            self.copy_from_server_to_local(remote_name, local_name)
            log.info(common_msg + "File %s has been locally cached to %s" %
                     (remote_name, local_name))
        elif os.path.getmtime(remote_name) > os.path.getmtime(local_name):
            log.warning(common_msg +
                        "File %s in cache will not be used: The remote file "
                        "(modified %s) is newer than the locally cached file "
                        "%s (modified %s)."
                        % (remote_name,
                           time.strftime(
                               '%Y-%m-%d %H:%M:%S',
                               time.localtime(os.path.getmtime(remote_name))
                           ),
                           local_name,
                           time.strftime(
                               '%Y-%m-%d %H:%M:%S',
                               time.localtime(os.path.getmtime(local_name))
                           )))
            self.release_writelock()
            return filename
        elif os.path.getsize(local_name) != os.path.getsize(remote_name):
            log.warning(common_msg +
                        "File %s not cached: The remote file (%d bytes) is of "
                        "a different size than the locally cached file %s "
                        "(%d bytes). The local cache might be corrupt."
                        % (remote_name, os.path.getsize(remote_name),
                           local_name, os.path.getsize(local_name)))
            self.release_writelock()
            return filename
        elif not os.access(local_name, os.R_OK):
            log.warning(common_msg +
                        "File %s in cache isn't readable. We will use the"
                        " remote version. Manually fix the permission."
                        % (local_name))
            self.release_writelock()
            return filename
        else:
            log.debug("File %s has previously been locally cached to %s" %
                      (remote_name, local_name))

        # Obtain a readlock on the downloaded file before releasing the
        # writelock. This is to prevent having a moment where there is no
        # lock on this file which could give the impression that it is
        # unused and therefore safe to delete.
        self.get_readlock(local_name)
        self.release_writelock()

        return local_name

    def copy_from_server_to_local(self, remote_fname, local_fname):
        """
        Copies a remote file locally

        Parameters
        ----------
        remote_fname : string
            Remote file to copy
        local_fname : string
            Path and name of the local copy to be made of the remote
            file.
        """

        head, tail = os.path.split(local_fname)
        head += os.path.sep
        if not os.path.exists(head):
            os.makedirs(os.path.dirname(head))

        command = 'cp ' + remote_fname + ' ' + local_fname
        os.system(command)
        # Copy the original group id and file permission
        st = os.stat(remote_fname)
        os.chmod(local_fname, st.st_mode)
        # If the user have read access to the data, but not a member
        # of the group, he can't set the group. So we must catch the
        # exception. But we still want to do this, for directory where
        # only member of the group can read that data.
        try:
            os.chown(local_fname, -1, st.st_gid)
        except OSError:
            pass

        # Need to give group write permission to the folders
        # For the locking mechanism
        # Try to set the original group as above
        dirs = os.path.dirname(local_fname).replace(self.dataset_local_dir, '')
        sep = dirs.split(os.path.sep)
        if sep[0] == "":
            sep = sep[1:]
        for i in range(len(sep)):
            orig_p = os.path.join(self.dataset_remote_dir, *sep[:i + 1])
            new_p = os.path.join(self.dataset_local_dir, *sep[:i + 1])
            orig_st = os.stat(orig_p)
            new_st = os.stat(new_p)
            if not new_st.st_mode & stat.S_IWGRP:
                os.chmod(new_p, new_st.st_mode | stat.S_IWGRP)
            if orig_st.st_gid != new_st.st_gid:
                try:
                    os.chown(new_p, -1, orig_st.st_gid)
                except OSError:
                    pass

    def disk_usage(self, path):
        """
        Return free usage about the given path, in bytes

        Parameters
        ----------
        path : string
            Folder for which to return disk usage

        Returns
        -------
        output : tuple
            Tuple containing total space in the folder and currently
            used space in the folder
        """

        st = os.statvfs(path)
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        return total, used

    def check_enough_space(self, remote_fname, local_fname,
                           max_disk_usage=0.9):
        """
        Check if the given local folder has enough space to store
        the specified remote file

        Parameters
        ----------
        remote_fname : string
            Path to the remote file
        remote_fname : string
            Path to the local folder
        max_disk_usage : float
            Fraction indicating how much of the total space in the
            local folder can be used before the local cache must stop
            adding to it.

        Returns
        -------
        output : boolean
            True if there is enough space to store the remote file.
        """

        storage_need = os.path.getsize(remote_fname)
        storage_total, storage_used = self.disk_usage(self.dataset_local_dir)

        # Instead of only looking if there's enough space, we ensure we do not
        # go over max disk usage level to avoid filling the disk/partition
        return ((storage_used + storage_need) <
                (storage_total * max_disk_usage))

    def safe_mkdir(self, folderName, force_perm=None):
        """
        Create the specified folder. If the parent folders do not
        exist, they are also created. If the folder already exists,
        nothing is done.

        Parameters
        ----------
        folderName : string
            Name of the folder to create
        force_perm : mode to use for folder creation
        """
        if os.path.exists(folderName):
            return
        intermediaryFolders = folderName.split(os.path.sep)

        # Remove invalid elements from intermediaryFolders
        if intermediaryFolders[-1] == "":
            intermediaryFolders = intermediaryFolders[:-1]
        if force_perm:
            force_perm_path = folderName.split(os.path.sep)
            if force_perm_path[-1] == "":
                force_perm_path = force_perm_path[:-1]
            base = len(force_perm_path) - len(intermediaryFolders)

        for i in range(1, len(intermediaryFolders)):
            folderToCreate = os.path.sep.join(intermediaryFolders[:i + 1])

            if os.path.exists(folderToCreate):
                continue
            os.mkdir(folderToCreate)
            if force_perm:
                os.chmod(folderToCreate, force_perm)

    def get_readlock(self, path):
        """
        Obtain a readlock on a file

        Parameters
        ----------
        path : string
            Name of the file on which to obtain a readlock
        """

        timestamp = int(time.time() * 1e6)
        lockdirName = "%s.readlock.%i.%i" % (path, self.pid, timestamp)
        os.mkdir(lockdirName)

        # Register function to release the readlock at the end of the script
        atexit.register(self.release_readlock, lockdirName=lockdirName)

    def release_readlock(self, lockdirName):
        """
        Release a previously obtained readlock

        Parameters
        ----------
        lockdirName : string
            Name of the previously obtained readlock
        """

        # Make sure the lock still exists before deleting it
        if (os.path.exists(lockdirName) and os.path.isdir(lockdirName)):
            os.rmdir(lockdirName)

    def get_writelock(self, filename):
        """
        Obtain a writelock on a file.
        Only one write lock may be held at any given time.

        Parameters
        ----------
        filename : string
            Name of the file on which to obtain a writelock
        """

        # compilelock expect locks to be on folder. Since we want a lock on a
        # file, we will have to ask compilelock for a folder with a different
        # name from the file we want a lock on or else compilelock will
        # try to create a folder with the same name as the file
        compilelock.get_lock(filename + ".writelock")

    def release_writelock(self):
        """
        Release the previously obtained writelock
        """
        compilelock.release_lock()


datasetCache = LocalDatasetCache()
