#!/usr/bin/env python
# -*- coding: utf-8

########################################
#
#
# This file is intentionally monolithic.
# It also intentionally restricts itself
# to standard library modules, with no
# extra dependencies.
#
from __future__ import print_function

__authors__   = "Steven Pigeon"
__copyright__ = "(c) 2012, Université de Montréal"
__contact__   = "Steven Pigeon: pigeon@iro.umontreal.ca"
__version__   = "dataset-get 0.1"
__licence__   = "BSD 3-Clause http://www.opensource.org/licenses/BSD-3-Clause "

import logging
import re,os,sys,shutil,time
import warnings
import urllib,urllib2
import tarfile
import subprocess

from theano.compat.six.moves import input

logger = logging.getLogger(__name__)


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


########################################
#
# Global variables for the whole module.
#
dataset_sources="sources.lst"
dataset_web="http://www.stevenpigeon.org/secret"
dataset_conf_path=""
dataset_data_path=""
root_conf_path=None
root_data_path=None
user_conf_path=None
user_data_path=None
super_powers=False

# both dictionaries for fast search
# (but are semantically lists)
packages_sources={}
installed_packages_list={}

########################################
def local_path_as_url( filename ):
    """
    Takes a local, OS-specific path or
    filename and transforms it into an
    url starting with file:// (it
    simplifies a lot of things).

    :param filename: a relative or absolute pathname
    :returns: the urlified absolute path
    """
    return "file://"+urllib.pathname2url(os.path.abspath(filename))

########################################
def has_super_powers():
    """
    Determines whether or not the program
    is run as root.

    :returns: true if run as root, false otherwise
    """
    return os.geteuid()==0

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
def get_timestamp_from_url( url ):
    """
    Gets the Last-Modified field from the
    http header associated with the file
    pointed to by the url. Raises whatever
    exception urllib2.urlopen raises.

    It can't lookup local file, unless they
    are presented as a file:/// url.

    :param url: a filename or an url
    :returns: the last-modified timestamp
    """

    obj = urllib2.urlopen( url )

    return time.strptime(
        obj.info()["Last-Modified"],
        "%a, %d %b %Y %H:%M:%S GMT") # RFC 2822 date format

########################################
def download_from_url( url, filename=None, progress_hook=None ):
    """
    Downloads a file from an URL in the
    specificed filename (or, if filename
    is None, to a temporary location).
    Returns the location of the downloaded
    file.

    :param url: url of the file to download
    :param filename: filename to download to (None means a temp file is created)
    :param progress_hook: a download hook to display progress
    :returns: the filename where the file was downloaded
    """
    (temp_filename, headers)=urllib.urlretrieve( url,filename,progress_hook )

    return temp_filename


########################################
def file_access_rights( filename, rights, check_above=False ):
    """
    Determines if a file has given rights.
    If the file exists, it tests for its
    rights. If it does not exist, and
    check_above=True, then it checks for
    the directory's rights, to test for
    write/creation rights.

    :param filename: filename of the file to assess
    :param rights: rights to be tested
    :param check_above: Check directory rights if file does not exist.
    :returns: boolean, whether 'rights' rights are OK
    """
    if os.path.exists(filename):
        return os.access(filename, rights)
    else:
        if check_above:
            return os.access(os.path.dirname(os.path.abspath(filename)), rights)
        else:
            return False



########################################
def atomic_replace( src_filename, dst_filename ):
    """
    Does an "atomic" replace of a file by another.

    If both files reside on the fame FS
    device, atomic_replace does a regular
    move. If not, the source file is first
    copied to a temporary location on the
    same FS as the destination, then a
    regular move is performed.

    caveat: the destination FS must have
    enough storage left for the temporary
    file.

    :param src_filename: The file to replace from
    :param dst_filename: The file to be replaced
    :raises: whatever shutil raises
    """

    ####################################
    def same_fs( filename_a, filename_b):
        """
        Checks if both files reside on the
        same FS device
        """
        stats_a = os.stat(filename_a)
        stats_b = os.stat(filename_b)
        return stats_a.st_dev == stats_b.st_dev;

    if os.path.exists(dst_filename) and not same_fs(src_filename,dst_filename):
        # deals with non-atomic move
        #
        dst_path = os.path.dirname(os.path.abspath(dst_filename))
        dst_temp_filename=os.tempnam(dst_path);
        shutil.copy(src_filename, dst_temp_filename) # non-atomic
        shutil.move(dst_temp_filename,dst_filename)  # atomic
    else:
        # an atomic move is performed
        # (both files are on the same device,
        # or the destination doesn't exist)
        #
        shutil.move(src_filename, dst_filename)




########################################
def set_defaults():
    """
    Detects whether the program is run
    as an ordinary user or as root, and
    then sets defauts directories for
    packages, configurations, and sources.

    caveat: this is an FreeDesktop-friendly
    version, and we will need eventually
    to have Windows- and OSX-friendly
    versions.

    See: http://freedesktop.org/wiki/Home
    and: http://www.linuxfoundation.org/collaborate/workgroups/lsb/fhs
    """

    global dataset_conf_path, \
           dataset_data_path, \
           root_conf_path, \
           root_data_path, \
           user_conf_path, \
           super_powers

    # a conspicuously LINUX version
    # (on windows, if we ever do a
    # windows version, these would
    # be different, and we may even
    # not have 'root' per se.)
    #
    root_conf_path="/etc/pylearn/"
    root_data_path="/usr/share/pylearn/dataset/"
    user_conf_path=os.path.join(os.environ["HOME"],".local/share/pylearn/")
    user_data_path=os.path.join(os.environ["HOME"],".local/share/pylearn/dataset/")

    if has_super_powers():
        dataset_conf_path=root_conf_path
        dataset_data_path=root_data_path
        super_powers=True
    else:
        dataset_conf_path=user_conf_path
        dataset_data_path=user_data_path
        super_powers=False


    # check if directories exist, and if not,
    # create them, and then fetch source.lst
    #
    if not os.path.exists(dataset_conf_path):
        os.makedirs(dataset_conf_path)

    if not os.path.exists(os.path.join(dataset_conf_path,dataset_sources)):
        atomic_update(os.path.join(dataset_web,dataset_sources),
                      os.path.join(dataset_conf_path,dataset_sources),
                      progress_bar)

    if not os.path.exists(dataset_data_path):
        os.makedirs(dataset_data_path)

    read_packages_sources()
    read_installed_packages_list();

########################################
def read_packages_sources():
    """
    Reads the sources.lst file and
    populates the available packages
    list.

    caveat: parsing of the sources.lst
    is pathetic

    Assigns: packages_sources
    :raises: RuntimeError if sources.lst cannot be read
    """

    def read_from_file(config_filename):
        """
        Reads a sources.lst file from a given location

        :param config_filename: the configuration file to read
        """

        global packages_sources
        try:
            f=open(config_filename,"r")
        except Exception as e:
            # not a problem if not found in a given location
            pass
        else:
            # file opened
            for line in f:
                t=line.rstrip().split(' ') # rstrips strips whitespaces at the end (\n)
                packages_sources[t[0]]=\
                    this_package=package_info(
                    config_filename,
                    t[0], # name
                    t[1], # timestamp
                    t[2], # human-readable size
                    urllib.unquote(t[3]), # source on the web
                    None)  # None as not installed (from source) (may be overridden later)

    if super_powers:
        read_from_file(os.path.join(dataset_conf_path,dataset_sources))
    else:
        # read root, user, then paths.
        paths=[ os.path.join(root_conf_path,dataset_sources),
                os.path.join(user_conf_path,dataset_sources) ]
        try:
            paths+=[ os.path.join(x,dataset_sources) for x in re.split(":|;",os.environ["PYLEARN2_DATA_PATH"]) ]
        except Exception:
            # PYLEARN2_DATA_PATH may or mayn't be defined
            pass

    for path in paths:
        read_from_file(path)
    if len(packages_sources)==0:
        raise RuntimeError( "[cf] fatal: could not find/read sources.lst (unexpected!)" )


########################################
def read_installed_packages_list():
    """
    Reads the various installed.lst files
    found on the system. First it searches
    for the root-installed installed.lst,
    then the user's, then searches the
    locations specified by the environment
    variable PYLEARN2_DATA_PATH (which is
    a standard :-separated list of locations)

    Assigns: installed_packages_list
    """
    # note: we add and overwrite rather
    # than clearing and filling (so we can
    # read many installed.lst, but the last
    # ones read overrides the earlier ones)
    #


    def read_from_file(config_filename):
        """
        Reads an installed.lst file from a given location

        :param config_filename: the configuration file to read
        """

        global installed_packages_list
        try:
            installed_list_file=open(config_filename)
        except IOError:
            # not a problem if not found in a location
            pass
        else:
            # read from file and
            # create a dictionary
            #
            for line in installed_list_file:
                l=line.rstrip().split(' ') # removes trailing whitespaces (\n)

                if l:
                    installed_packages_list[l[0]]=\
                        this_package=package_info(
                        config_filename,
                        l[0], # name
                        l[1], # timestamp
                        l[2], # human-readable size
                        urllib.unquote(l[3]), # source on the web
                        urllib.unquote(l[4]))  # where installed
                else:
                    pass # skip blank lines (there shouldn't be any)


    if super_powers:
        # then read only root
        read_from_file(os.path.join(dataset_conf_path,"installed.lst"))
    else:
        # read root, user, then paths.
        paths=[ os.path.join(root_conf_path,"installed.lst"),
                os.path.join(user_conf_path,"installed.lst") ]
        try:
            paths+=[ os.path.join(x,"installed.lst") for x in re.split(":|;",os.environ["PYLEARN2_DATA_PATH"]) ]
        except Exception:
            # PYLEARN2_DATA_PATH may or mayn't be defined
            pass

    for path in paths:
        read_from_file(path)
    if len(installed_packages_list)==0:
        logger.warning("[cf] no install.lst found "
                       "(will be created on install/upgrade)")


########################################
def write_installed_packages_list():
    """
    Saves the installed package list and
    their location (file over-writen depends
    on run as root or as a normal user)
    """
    global installed_packages_list
    try:
        tmp=open(os.path.join(dataset_conf_path,"installed.lst.2"),"w")
    except IOError:
        raise RuntimeError("[cf] fatal: cannot create temp file")
    else:
        # ok, probably worked?
        for package in installed_packages_list.values():
            # adds only packages that are readable for
            # this user (maybe some site-installed datasets
            # are out of his reach)
            #
            if package.where!=None and \
                    file_access_rights(os.path.join(package.where,package.name),
                                      os.F_OK | os.R_OK):
                print(
                    " ".join(map(str,[ package.name,
                                       package.timestamp,
                                       package.readable_size,
                                       urllib.quote(package.source,"/:~"),
                                       urllib.quote(package.where,"/:~") ] )),
                    file=tmp)

        # replace the installed.lst in
        # a safe way
        atomic_replace(os.path.join(dataset_conf_path,"installed.lst.2"),
                       os.path.join(dataset_conf_path,"installed.lst"))


########################################
def atomic_update( remote_src, local_dst, hook=None ):
    """
    Takes a (possibly) remote file an checks
    if it is newer than a(n obligatoritly)
    local file. If the source is newer, an
    "atomic update" is performed.

    Atomic here means that the source is
    downloaded in a distinct location, and
    only if download is successful is the
    destination file replaced atomically.

    :param remote_src: Url to a (possibly) remote file
    :param local_dst: file to update
    :param hook: download progress hook
    :raises: various IOErrors
    """

    global hook_download_filename # hook-related

    try:
        remote_date = get_timestamp_from_url(remote_src);
    except IOError as e:
        raise IOError("[ts] %s %s" % (str(e),remote_src))
    else:
        if os.path.exists(local_dst):
            # it exists, test for update
            try:
                local_date = get_timestamp_from_url(local_path_as_url(local_dst))
            except Exception as e:
                raise IOError("[ts] %s %s" % (str(e),local_dst))
            else:
                if (local_date<remote_date):
                    # OK, the file seems to be out-of-date
                    # let's update it
                    #
                    if file_access_rights(local_dst,os.W_OK,check_above=True):
                        # we have write access to the file, or if it doesn't
                        # exist, to the directory where we want to write it.
                        #
                        try:
                            hook_download_filename=remote_src # hook-related
                            temp_filename=download_from_url(remote_src, filename=None, progress_hook=hook)
                        except Exception as e:
                            raise IOError("[dl] %s %s" % (str(e),remote_src))
                        else:
                            # download to temporary was successful,
                            # let's (try to) perform the atomic replace
                            #
                            try:
                                atomic_replace(temp_filename,local_dst)
                            except Exception as e:
                                raise IOError("[ac] %s %s --> %s" % (str(e),temp_filename,local_dst))
                    else:
                        raise IOError("[rw] no write access to %s " % local_dst )
                else:
                    # file's up to date, everything's fine
                    # and there's nothing else to do
                    #
                    pass
        else:
            # file does not exist, just download!
            #
            if file_access_rights(local_dst,os.W_OK,check_above=True):

                try:
                    hook_download_filename=remote_src # hook-related
                    temp_filename=download_from_url(remote_src, filename=None, progress_hook=hook)
                except Exception as e:
                    raise IOError("[dl] %s %s" % (str(e),remote_src))
                else:
                    # yay! download successful!
                    #
                    try:
                        atomic_replace(temp_filename,local_dst)
                    except Exception as e:
                        raise IOError("[ac] %s %s --> %s" % (str(e),temp_filename,local_dst))
            else:
                raise IOError("[rw] no right access to %s" % local_dst)


########################################
def unpack_tarball( tar_filename, dest_path ):
    """
    Unpacks a (bzipped2) tarball to a destination
    directory

    :param tar_filename: the bzipped2 tar file
    :param dest_path: a path to where expand the tarball
    :raises: various IOErrors
    """

    if os.path.exists(tar_filename):
        if file_access_rights(dest_path,os.W_OK,check_above=False):
            try:
                # open the tarball as read, bz2
                #
                this_tar_file=tarfile.open(tar_filename,"r:bz2")
            except Exception as e:
                raise IOError("[tar] cannot open '%s'" % tar_filename)
            else:
                # ok, it's openable, let's expand it
                #
                try:
                    this_tar_file.extractall(dest_path)
                except Exception as e:
                    raise IOError("[tar] error while extracting '%s'" %tar_filename)
                else:
                    # yay! success!
                    pass
        else:
            raise IOError("[tar] no right access to '%s'" % dest_path)

    else:
        raise IOError("'%s' not found" % tar_filename)


########################################
def run_scripts( package_location, scripts ):
    """
    Search for installation scripts speficied
    by the scripts list

    :param package_location: "root" path for the package
    :param scripts: list of scripts to look for (and execute)
    :raises: subprocess exceptions
    """

    path=os.path.join(package_location,"scripts/")

    cwd=os.getcwd()
    os.chdir(path)

    for script in scripts:

        if os.path.exists( script ):
            # throws CalledProcessError if return
            # return code is not zero.
            #
            try:
                subprocess.check_call( script, stdout=sys.stdout, stderr=sys.stderr  )
            except Exception:
                os.chdir(cwd)
                raise

    # ok, success (or not), let's unstack
    os.chdir(cwd)


########################################
def install_package( package, src, dst ):
    """
    Unpacks a (bzipped2) tarball and
    expands it to the given location.

    If unpacking is successful, installation
    scripts are run.

    :param package: package information
    :param src: the source tarball
    :param dst: the destination directory
    :raises: IOErrors and subprocess exceptions
    """

    #FIXME: change creation flags to group-public
    #       readable when invoked with super-powers
    #
    unpack_tarball(src,dst)
    run_scripts(dst+package.name, scripts=["getscript","postinst"] )


########################################
def remove_package(package,dst):
    """
    Removes a script by running the
    various removal scripts, then by
    deleting files and directories.

    :param package: package information
    :param dst: packages root (where packages are installed)
    """

    #path=os.path.join(dst,package.name)
    path=os.path.join(package.where,package.name)
    #print path

    run_scripts(path,scripts=["prerm"])
    shutil.rmtree(os.path.join(path,"data/"))
    shutil.rmtree(os.path.join(path,"docs/"))

    run_scripts(os.path.join(dst,package.name),scripts=["postrm"])
    shutil.rmtree(os.path.join(path,"scripts/"))
    shutil.rmtree(path)

    update_installed_list("r",package)




########################################
def update_installed_list( op, package ):
    """
    Updates the internal list of installed
    packages. The operation is either "i"
    for install and update, or "r" for removal

    :param op: the operation performed
    :param package: the package information
    :param dst: where the package was installed
    """

    if op=="i":
        installed_packages_list[package.name]=package;
    elif op=="r":
        # remove the package from the list
        del installed_packages_list[package.name]
    else:
        raise RuntimeError("[cf] fatal: invalid configuration op '%s'." % op)

    write_installed_packages_list()


########################################
def show_packages():
    """
    List all available packages, both
    installed or from remove sources
    """
    logger.info("These packages are available:")
    for this_package in packages_sources.values():
        if this_package.name in installed_packages_list:
            state="u" if installed_packages_list[this_package.name].timestamp<this_package.timestamp else 'i';
        else:
            state="-"
        package_time = time.strftime("%a, %d %b %Y %H:%M:%S",
                                     time.gmtime(this_package.timestamp))

        logger.info("{0} {1:<20} {2:<8} "
                    "{3:<30} {4}".format(state,
                                         this_package.name,
                                         this_package.readable_size,
                                         package_time,
                                         this_package.source))

########################################
def install_upgrade( package, upgrade=False,progress_hook=None ):
    """
    This function installs or upgrades a package.

    :param package: package information
    :param upgrade: If True, performs and upgrade, installs underwise
    :param progress_hook: a download progress hook
    """

    global hook_download_filename # hook-related

    if upgrade:
        operation = "[up] upgrading"
    else:
        operation = "[in] installing"
    logger.info("{0} '{1}' to {2}".format(operation,
                                          package.name, dataset_data_path))


    remote_src=package.source

    # install location is determined by super-powers
    # (so a root package can be upgraded locally!)
    package.where=dataset_data_path;

    # TODO: to add caching, first lookup the
    # tarball in the package cache (but there's'nt
    # one for now)
    #
    cached=False;

    if not cached:
        hook_download_filename=remote_src # hook-related
        temp_filename=download_from_url(remote_src,filename=None,progress_hook=progress_hook)
    else:
        # assign filename to cached package
        pass

    logger.info("[in] running install scripts "
                "for package '{0}'".format(package.name))

    # runs through the .../package_name/scripts/
    # directory and executes the scripts in a
    # specific order (which shouldn't display
    # much unless they fail)
    #
    install_package(package,temp_filename,dataset_data_path)
    update_installed_list("i",package)



########################################
def upgrade_packages(packages_to_upgrade, hook=None ):
    """
    Upgrades packages.

    If no packages are supplied, it will perform
    an "update-all" operation, finding all packages
    that are out of date.

    If packages names are supplied, only those
    are checked for upgrade (and upgraded if out
    of date)

    :param packages_to_upgrade: list of package names.
    :raises: IOErrors (from downloads/rights)
    """

    # get names only
    if packages_to_upgrade==[]:
        packages_to_upgrade=installed_packages_list.keys() # all installed!
        all_packages=True
    else:
        all_packages=False

    # check what packages are in the list,
    # and really to be upgraded.
    #
    packages_really_to_upgrade=[]
    for this_package in packages_to_upgrade:
        if this_package in installed_packages_list:

            # check if there's a date
            installed_date=installed_packages_list[this_package].timestamp

            if this_package in packages_sources:
                repo_date=packages_sources[this_package].timestamp

                if installed_date < repo_date:
                    # ok, there's a newer version
                    logger.info(this_package)
                    packages_really_to_upgrade.append(this_package)
                else:
                    # no newer version, nothing to update
                    pass
            else:
                logger.warning("[up] '{0}' is unknown "
                               "(installed from file?).".format(this_package))
        else:
            # not installed?
            if not all_packages:
                logger.warning("[up] '{0}' is not installed, "
                               "cannot upgrade.".format(this_package))
                pass


    # once we have determined which packages
    # are to be updated, we show them to the
    # user for him to confirm
    #
    if packages_really_to_upgrade!=[]:
        logger.info("[up] the following package(s) will be upgraded:")
        for this_package in packages_really_to_upgrade:
            readable_size = packages_sources[this_package].readable_size
            logger.info("{0} ({1})".format(this_package, readable_size))

        r = input("Proceed? [yes/N] ")
        if r=='y' or r=='yes':
            for  this_package in packages_really_to_upgrade:
                install_upgrade( packages_sources[this_package], upgrade=True, progress_hook=hook )
        else:
            logger.info("[up] Taking '{0}' for no, so there.".format(r))
    else:
        # ok, nothing to upgrade,
        # move along.
        pass



########################################
#
# installs the packages, and forces if
# they already exist
#
# packages must be supplied as argument.
#
#
def install_packages( packages_to_install, force_install=False, hook=None ):
    """
    Installs the packages, possibly forcing installs.

    :param packages_to_install: list of package names
    :param force_install: if True, re-installs even if installed.
    :param hook: download progress hook
    :raises: IOErrors
    """

    if packages_to_install==[]:
        raise RuntimeError("[in] fatal: need packages names to install.")

    if force_install:
        logger.warning("[in] using the force")

    packages_really_to_install=[]
    for this_package in packages_to_install:
        if this_package in packages_sources:

            if force_install or not this_package in installed_packages_list:
                packages_really_to_install.append(this_package)
            else:
                logger.warning("[in] package '{0}' "
                               "is already installed".format(this_package))
        else:
            logger.warning("[in] unknown package '{0}'".format(this_package))

    if packages_really_to_install!=[]:
        logger.info("[in] The following package(s) will be installed:")
        for this_package in packages_really_to_install:
            readable_size = packages_sources[this_package].readable_size
            logger.info("{0} ({1})".format(this_package, readable_size))

        r = input("Proceed? [yes/N] ")
        if r=='y' or r=='yes':
            for  this_package in packages_really_to_install:
                install_upgrade( packages_sources[this_package], upgrade=False, progress_hook=hook )
        else:
            logger.info("[in] Taking '{0}' for no, so there.".format(r))
    else:
        # ok, nothing to upgrade,
        # move along.
        pass



########################################
def install_packages_from_file( packages_to_install ):
    """
    (Force)Installs packages from files, but does
    not update installed.lst files.

    caveat: not as tested as everything else.

    :param packages_to_install: list of files to install
    :raises: IOErrors
    """
    if packages_to_install==[]:
        raise RuntimeError("[in] fatal: need packages names to install.")

    packages_really_to_install=[]
    for this_package in packages_to_install:
        if os.path.exists(this_package):
            packages_really_to_install.append(this_package)
        else:
            logger.warning("[in] package '{0}' not found".format(this_package))

    if packages_really_to_install!=[]:
        logger.info("[in] The following package(s) will be installed:")
        packages = []
        for this_package in packages_really_to_install:
            packages.append(corename(this_package))
        logger.info(' '.join(packages))

        r = input("Proceed? [yes/N] ")
        if r=='y' or r=='yes':
            for  this_package in packages_really_to_install:
                #install_upgrade( this_package, upgrade=False, progress_hook=hook )
                if os.path.exists(dataset_data_path+corename(this_package)):
                    r = input("[in] '%s' already installed, overwrite? [yes/N] " % corename(this_package))

                    if r!='y' and r!='yes':
                        logger.info("[in] skipping package "
                                    "'{0}'".format(corename(this_package)))
                        continue
                install_package( corename(this_package), this_package, dataset_data_path)
                #update_installed_list("i",(make a package object here),dataset_data_path)

        else:
            logger.info("[in] Taking '{0}' for no, so there.".format(r))




########################################
#
# uninstall packages, whether or not they
# are found in the sources.lst file (to
# account for the packages installed from
# file)
#
# like install, it expects a list, if there's
# no list, nothing happens. It will test
# whether or not the packages are installed, and
# will ask the user for a confirmation.
#
def remove_packages( packages_to_remove ):
    """
    Uninstall packages, whether or not they
    are found in the source.lst (so it can
    remove datasets installed from file).

    :param packages_to_remove: list of package names
    :raises: IOErrors
    """

    if packages_to_remove==[]:
        raise RuntimeError("[rm] fatal: need packages names to remove.")

    packages_really_to_remove=[]
    for this_package in packages_to_remove:
        if this_package in packages_sources:

            #this_data_set_location=os.path.join(dataset_data_path,this_package)

            # check if in the installed.lst
            # then if directory actually exists
            # then if you have rights to remove it
            if this_package in installed_packages_list:

                this_data_set_location=os.path.join( installed_packages_list[this_package].where,
                                                     installed_packages_list[this_package].name )

                if os.path.exists(this_data_set_location):
                    if (file_access_rights(this_data_set_location,os.W_OK)):
                        # ok, you may have rights to delete it
                        packages_really_to_remove.append(this_package)
                    else:
                        logger.warning("[rm] insufficient rights "
                                       "to remove '{0}'".format(this_package))
                else:
                    logger.warning("[rm] package '{0}' found in config file "
                                   "but not installed".format(this_package))
            else:
                logger.warning("[rm] package '{0}' "
                               "not installed".format(this_package))
        else:
            logger.warning("[rm] unknown package '{0}'".format(this_package))

    if packages_really_to_remove!=[]:
        logger.info("[rm] the following packages will be removed permanently:")
        packages = []
        for this_package in packages_really_to_remove:
            packages.append(this_package)
        logger.info(' '.join(packages))

        r = input("Proceed? [yes/N] ")
        if r=='y' or r=='yes':
            for  this_package in packages_really_to_remove:
                remove_package( installed_packages_list[this_package], dataset_data_path )
        else:
            logger.info("[up] Taking '{0}' for no, so there.".format(r))
    else:
        # ok, nothing to remove, filenames where bad.
        pass




########################################
hook_download_filename=""
def progress_bar( blocks, blocksize, totalsize ):
    """
    Simple hook to show download progress.

    caveat: not that great-looking, fix later to
            a cooler progress bar or something.
    """
    print("\r[dl] %6.2f%% %s" % (min(totalsize,blocks*blocksize)*100.0/totalsize, hook_download_filename), end='')
    sys.stdout.flush()



########################################
def process_arguments():
    """
    Processes the installation arguments (from
    the command line)

    The possible arguments are:

    list
         lists available datasets from
         sources.lst

    update
         updates sources.lst

    upgrade
         upgrades datasets that are out
         of date

    install <dataset1> <dataset2> ... <datasetn>
         uses sources.lst to locate the
         package and perform the installation

     force-install <dataset1> ... <datasetn>
         performs an install even if the data
         sets seem to be there.

     remove <dataset1> <dataset2> ... <datasetn>
         removes the dataset

     clean
         empties package cache (does nothing
         for now, because no cache.)
    """

    if len(sys.argv)>1:

        # due to the relative simplicity of the
        # arguments, we won't use optparse (2.3-2.6)
        # nor argparse (2.7+), although in the future
        # it may pose problems

        if sys.argv[1]=="list":
            show_packages()

        elif sys.argv[1]=="update":
            atomic_update( os.path.join(dataset_web,dataset_sources),
                           os.path.join(dataset_conf_path,dataset_sources),
                           hook=progress_bar)

        elif sys.argv[1]=="upgrade":
            upgrade_packages(sys.argv[2:],
                             hook=progress_bar)

        elif sys.argv[1]=="install":
            install_packages(sys.argv[2:],
                             hook=progress_bar)

        elif sys.argv[1]=="install-from-file":
            install_packages_from_file(sys.argv[2:])

        elif sys.argv[1]=="force-install":
            install_packages(sys.argv[2:],
                             force_install=True,
                             hook=progress_bar)

        elif sys.argv[1]=="remove":
            remove_packages(sys.argv[2:])

        elif sys.argv[1]=="clean":
            # does nothing, no cache implemented
            # yet.
            pass


        elif sys.argv[1]=="version":
            logger.info(__version__)

        else:
            raise RuntimeError("[cl] unknown command '%s'" % sys.argv[1])
    else:
        raise RuntimeError("[cl] missing command")



########################################
if __name__ == "__main__":
    # to remove RuntimeWarnings about how
    # tempfilename is unsafe.
    #
    warnings.simplefilter("ignore", RuntimeWarning)

    # OK, let's construct the environment
    # needed by dataset-get
    try:
        set_defaults()
    except Exception as e:
        logger.exception(e)
        exit(1) # fail!

    try:
        process_arguments()
    except Exception as e:
        logger.exception(e)
        exit(1)
