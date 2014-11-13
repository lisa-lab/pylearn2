#!/usr/bin/env python

"""A simple resolution mechanism to find datasets"""

import logging
import re,os,urllib

logger = logging.getLogger(__name__)


class dataset_resolver:

    ########################################
    class package_info:
        """
        A simple class to structure
        the package's information
        """
        def __init__(self, cf, name,ts,rs,src,whr):
            self.configuration_file=cf # in which configuration file was it found?
            self.name=name         # the short name, e.g., "mnist"
            self.timestamp=int(ts) # a unix ctime
            self.readable_size=rs  # a human-readable size, e.g., "401.3MB"
            self.source=src        # the web source
            self.where=whr         # where on this machine



    installed_packages_list={}

    def read_installed_packages_list(self, from_location):
        """
        Reads a given configuration file, and updates the
        internal installed packages list.

        :param from_location: a path (string) to a directory containing an installed.lst file
        """

        try:
            installed_list_file=open(from_location+"/installed.lst")
        except IOError:
            # maybe not a problem, but
            # FIXME: print a warning if exists,
            # but cannot be read (permissions)
            pass
        else:
            # read from file and
            # create a dictionary
            for line in installed_list_file:
                l=line.rstrip().split(' ')
                if l:
                    self.installed_packages_list[l[0]]=\
                        this_package=self.package_info(
                        from_location, # from which configuration files it comes
                        l[0], # name
                        l[1], # timestamp
                        l[2], # human-readable size
                        urllib.unquote(l[3]), # source on the web
                        urllib.unquote(l[4]))  # where installed
                else:
                    pass# skip blank lines (there shouldn't be any)


    def resolve_dataset(self,dataset_name):
        """
        Looks up a dataset name and return its location,
        or None if it's unknown.

        :param dataset_name: a canonical dataset name, 'mnist' for e.g.
        :returns: A path, if dataset is found, or None otherwise.
        """
        if dataset_name in self.installed_packages_list:
            return os.path.join( self.installed_packages_list[dataset_name].where,
                                 self.installed_packages_list[dataset_name].name )
        else:
            return None


    def __init__(self):
        """
        Scans possible locations to load installed.lst files. It first
        scans the root install, the user install, then the paths, from left
        to right, specified by the PYLEARN2_DATA_PATH environment variable.
        """

        paths= ["/etc/pylearn/", os.environ["HOME"]+"/.local/share/pylearn/"]
        try:
            paths+=re.split(":|;",os.environ["PYLEARN2_DATA_PATH"])
        except Exception:
            # PYLEARN2_DATA_PATH may or mayn't be defined
            pass

        for path in paths:
            self.read_installed_packages_list(path)




if __name__=="__main__":
    # simplest tests
    x=dataset_resolver()
    logger.info(x.resolve_dataset("toaster-oven"))
    logger.info(x.resolve_dataset("fake-dataset"))
