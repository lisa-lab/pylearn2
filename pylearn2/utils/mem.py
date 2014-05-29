"""
.. todo::

    WRITEME
"""
import subprocess
import os


def get_memory_usage():
    """
    Return int containing memory used by this process. Don't trust this
    too much, I'm not totally sure what ps rss measures.
    """

    pid = os.getpid()
    process = subprocess.Popen("ps -o rss %s | awk '{sum+=$1} END {print sum}'" % pid,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    )
    stdout_list = process.communicate()[0].split('\n')
    return int(stdout_list[0])


def improve_memory_error_message(error, msg=""):
    """
    Raises a TypicalMemoryError if the MemoryError has no messages

    Parameters
    ----------
    error: MemoryError
        An instance of MemoryError
    msg: string
        A message explaining what possibly happened
    """
    assert isinstance(error, MemoryError)

    if str(error):
        raise error
    else:
        raise TypicalMemoryError(msg)


class TypicalMemoryError(MemoryError):
    """
    Memory error that could have been caused by typical errors such
    as using 32bit python while having more than 2-4GB or RAM computer.

    This is to help users understand what could have possibly caused the
    memory error. A more detailed explanation needs to be given for
    each case.

    Parameters
    ----------
    value: str
        String explaining what the program was trying to do and
        why the memory error possibly happened.
    """
    def __init__(self, value):
        super(TypicalMemoryError, self).__init__()

        # could add more typical errors to this string
        self.value = value + ("\n + Make sure you use a 64bit python version. "
                              "32bit python version can only access 2GB of "
                              "memory on windows and 4GB of memory on "
                              "linux/OSX.")

    def __str__(self):
        return repr(self.value)
