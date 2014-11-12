__author__ = 'denadai2'

from configparser import (AddPylearnConfigVar, SemicolonParam)

from theano.configparser import (TheanoConfigParser)


pylearn2_config = TheanoConfigParser()

AddPylearnConfigVar('channels_include',
                    "Iterable of patterns that match channels to monitor during the training process. "
                    "It works as Unix shell-style wildcards. "
                    "Defaults to '*'",
                    SemicolonParam('*'),
)

AddPylearnConfigVar('channels_exclude',
                    "Iterable of patterns that match channels to exclude monitoring during the training process. "
                    "It works as Unix shell-style wildcards. "
                    "Defaults to ''",
                    SemicolonParam(''),
)