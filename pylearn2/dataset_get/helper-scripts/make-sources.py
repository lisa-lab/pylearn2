#!/usr/bin/env python
# -*- coding: utf-8

__authors__   = "Steven Pigeon"
__copyright__ = "(c) 2012, Université de Montréal"
__contact__   = "Steven Pigeon: pigeon@iro.umontreal.ca"
__version__   = "make-sources 0.1"
__licence__   = "BSD 3-Clause http://www.opensource.org/licenses/BSD-3-Clause "


import logging
import sys,os

logger = logging.getLogger(__name__)


########################################
def corename( filename ):
    """
    returns the 'corename' of a file. For
    example, corename("thingie.tar.bz2")
    returns "thingie" (a totally correct
    way of doing this would be to use
    MIME-approved standard extensions, in
    order to distinguish from, say a file
    "thingie.tar.bz2" and another file
    "my.data.tar.bz2"---for which we would
    have only "my" as corename)

    :param filename: a (base) filename
    :returns: the "core" filename
    """

    f1=None
    f2=os.path.basename(filename)

    # repeatedly remove the right-most
    # extension, until none is found
    #
    while f1 != f2:
        f1=f2
        (f2,ext)=os.path.splitext(f1)

    return f2

########################################
def human_readable_size(size):
    """
    Returns an approximate, human-readable,
    file size.

    :param size: an integer-like object
    :returns: a human-readable size as a string
    :raises: RunetimeError if file size exceeds petabytes (or is negative)
    """
    if (size>=0):
        for x in ['B','KB','MB','GB','TB','PB']:
            if size<1024:
                return ("%3.1f%s" if x!='B' else "%d%s") % (size, x)
            size/=1024.0

    raise RuntimeError("file size suspiciously large")


########################################
def print_table(where):
    """
    Generates the sources.lst table from path 'where'

    :param where: a (simple) path where to look for archives
    """
    for this_file in os.listdir(where):
        if ".tar.bz2" in this_file:
            full_file = os.path.join(where,this_file)
            size = human_readable_size(os.stat(full_file).st_size)
            logger.info("{0} {1} {2} "
                        "{3}".format(corename(this_file),
                                     os.path.getctime(full_file),
                                     size,
                                     os.path.join(repos_root, this_file)))

########################################
if __name__=="__main__":
    repos_root=sys.argv[1]
    print_table(sys.argv[2] if len(sys.argv)>2 else ".")
