==================================
dataset-get experimental framework
==================================

WARNING: This is a prototype that don't have real datasets
available. We don't know if we will use it in the future and we don't
use it. So you are probably better to skip this.


This directory contains a prototype of the unified dataset repository
infrastructure for Theano/PyLearn


Manifesto (the short version)
-----------------------------

- Public/Open Datasets should be always available to the community

- Datasets should be simple to get

- Datasets should be available "forever"

- Datasets should be in platform-independent formats

- Dataset should be curated by the Benevolent Gods of Data in order to make
  experiments repeatable and standardized

- People should be able to install datasets easily either as normal users
  or as superusers

- Dataset should be easily usable by usual tools, in our case
  Theano/PyLearn (while maintaining compatibility with basic tools such as
  numpy)


Manifesto (the slightly longer version)
---------------------------------------

- It is not enough to have datasets circulated, we must have datasets that
  are demonstrably from the original source, and not some n-th generation
  dataset than went through a great number of transcodings/preprocessing.

- Canonical datasets should be gathered in a curated repository, where they
  can reside for an indefinite amount of time ("forever", as forever goes
  in this world). The user can just get the dataset from the repository and
  reinstall *the same dataset* each time, making experiments stable and
  reproducible. This will be the responsibility of the BGoDs.

- Curating datasets to platform-independent formats is also the
  responsibility of the BGoDs, but the onus may be devoted to the people
  submitting a new dataset.

- Configuration should be transparent to the user, letting the tool update,
  copy, move, configure datasets without having the user manually editing
  anything. The configuration and datasets have to mesh with the tools we
  use, whether PyLearn, Theano, or numpy-based applications (possibly, in
  this case, with helper classes).


Key Design Goals of the dataset-get tool
----------------------------------------

- Everyone can use it. Up to now, we have been relying on the awesome work
  of the BGoDs of our lab to configure the datasets for everyone to use,
  but that solution does not scale well to people outside our lab, or
  people inside our lab but with computers that can be disconnected from
  the lab's repository---also known as laptops. Furthermore, currently,
  installation of datasets requires manual interventions (finding a
  compatible version, downloading it, decompress it in a suitable place,
  change configuration files by hand, etc.) which really should abstracted
  and automated by a simple-to-use tool. Therefore, the tool must automate
  adding (and removing) datasets for the user.

- Privileged/unprivileged user. Being able to operate a tool in user-space
  is important for user using shared systems on which they have little or
  no privileges. The dataset-get tool is strongly inspired by Debian's
  apt-get tool, but it works quite differently whether you invoke it as
  root or as a normal user (especially that you cannot invoke apt-get as a
  normal user) [1].

  - Invoked as a super-user, dataset-get affects system-wide configuration
    files. It will update root-accessible configuration files, but will
    create readable datasets for all (or as defined by the wanted file
    creation flags). Root can affect only root configuration, and while
    root can add or remove datasets, normal users do not need to do
    anything in particular.

  - As a normal, unprivileged user, dataset-get affects only the user's
    files. The user will have access to system-wide installed datasets in
    addition to his own.

  - Super- and normal-user configuration must mesh. The configuration files
    are this read in the following order: root configuration, user
    configuration, then configurations at paths specified by the
    PYLEARN2_DATA_PATH environment variable (which will now become a :- or
    ;-separated list of paths). Each configuration read may overwrite some
    of the previously read configuration items. That is, if root has
    dataset X in location Y, but user has dataset X in location W, then the
    user only sees the dataset in location W.

  Furthermore, dataset-get should perform OS-specific and OS-friendly
  configuration. That is, on an OS complying to the Open Desktop
  initiative, configuration files will be created in places that respect
  the spirit of the platform. For now, only Open Desktop-like systems are
  supported (but we will include OS and Windows in future versions).

- Transparent to the programmer. The configuration files created by
  dataset-get should be transparently used by Theano/PyLearn. In PyLearn,
  one should use the resolver class to look-up the location of a dataset
  and use an appropriate dataset object to read it.

The dataset-get tool is still a work in progress, but vows to uphold all of
the above goals in the long term (some of it is already working as it
should, some, like Windows and OS support, are missing).


TO DO
-----

This is a work in progress. There are many things still to do:

- Create a stable, long-term repository for the dataset

- Add a configuration for alternate/mirror or even multiple repositories.

- Make an OS/X (and BSD?) version

- Make a Windows 7 version (support for Windows PX and Vista will not
  happen. Windows 8 should be supported as well)

- Establish a dataset hierarchy or different name spaces. Standard datasets
  should be found in one place, and paper-related or other specific,
  non-canonical datasets in other places; c.f. Debian hierarchy of
  repositories.



Refs
----

[1] Francois-Denis Gonthier, Steven Pigeon --- Non Privileged User Package
Management: Use Cases, Issues, Proposed Solutions --- Pro cs. Linux
Symposium, 2009, p. 111-122

http://www.stevenpigeon.org/Publications/publications/ols-2009.pdf 
