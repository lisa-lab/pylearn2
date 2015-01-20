#!/usr/bin/env python
# -*- coding: utf-8

__authors__   = "Steven Pigeon"
__copyright__ = "(c) 2012 Université de Montréal"
__contact__   = "Steven Pigeon: pigeon@iro.umontreal.ca"
__version__   = "make-archive 0.1"
__licence__   = "BSD 3-Clause http://www.opensource.org/licenses/BSD-3-Clause "


import logging
import os, sys, tarfile

from theano.compat.six.moves import input

logger = logging.getLogger(__name__)


########################################
def checks(path):
    """
    Checks if pretty much everything is
    there, aborts if mandatory elements
    are missing, warns if strongly
    suggested are not found.

    :param path: path to the root of the dataset
    :returns: True, if the archive passed the test, False otherwise.
    """
    # path,
    # m for mandatory,
    # o for optional
    # s for strongly suggested,
    #
    check_for=[ ("data/",'m'),
                ("docs/",'m'),
                ("docs/license.txt",'m'),
                ("scripts/",'m'),
                ("scripts/getscript",'o'),
                ("scripts/postinst",'o'),
                ("scripts/prerm",'o'),
                ("scripts/postrm",'o'),
                ("readme.1rst",'s')
                ]

    found=0
    for (filename,mode) in check_for:
        this_check=os.path.join(path,filename)
        if os.path.exists(this_check):
            if os.path.isdir(this_check):
                if len(os.listdir(this_check))==0:
                    logger.warning("directory '{0}' "
                                   "is empty.".format(this_check))
            found+=1;
        else:
            if mode=='m':
                # fatal
                logger.error("'{0}' not found "
                             "but mandatory".format(this_check))
                return False
            elif mode=='s':
                # benign
                logger.warning("no '{0}' found".format(this_check))
            else:
                # whatever
                pass
    return (found>0)

########################################
def create_archive( source, archive_name ):

    if os.path.exists(archive_name):
        r = input("'%s' exists, overwrite? [yes/N] " % archive_name)
        if (r!="y") and (r!="yes"):
            logger.info("taking '{0}' for no, so there.".format(r))
            #bail out
            return

    try:
        tar=tarfile.open(archive_name,mode="w:bz2")
    except Exception as e:
        logger.exception(e)
        return
    else:
        for root, dirs, files in os.walk(source):
            for filename in files:
                this_file = os.path.join(root,filename)
                logger.info("adding '{0}'".format(this_file))
                tar.add(this_file)
        tar.close()


if __name__=="__main__":
    filename=sys.argv[1]
    if checks(filename):
        basename=os.path.basename(filename)
        ext=".tar.bz2"
        archive_name=basename+ext
        logger.info("Creating Archive '{0}'".format(archive_name))
        create_archive(filename,archive_name)
    else:
        logger.info("nothing found, aborting.")
