import subprocess as sp
import sys


def run_shell_command(cmd):
    """ Runs cmd as a shell command.
    Waits for it to finish executing, then returns all output
    printed to standard error and standard out, and the return code
    """
    child = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    output = child.communicate()[0]
    rc = child.returncode
    return output, rc


def print_progression(percent, width=50, delimiters=['[', ']'], symbol='#'):
    """
    Prints a progress bar to the command line
    """
    n_symbols = int(percent/100.0*width)
    progress_bar = delimiters[0] + n_symbols * symbol \
                                 + (width - n_symbols) * ' ' \
                                 + delimiters[1] + " "
    sys.stdout.write("\r" + progress_bar + str(percent) + "%")
    sys.stdout.flush()
