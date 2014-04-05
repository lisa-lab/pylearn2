"""
Utilities for running shell scripts and interacting with the terminal
"""
import subprocess as sp
import sys


def run_shell_command(cmd):
    """
    Runs cmd as a shell command. Waits for it to finish executing,
    then returns all output printed to standard error and standard out,
    and the return code.

    Parameters
    ----------
    cmd : str
        The shell command to run

    Returns
    -------
    output : str
        The string output of the process
    rc : WRITEME
        The numeric return code of the process
    """
    child = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    output = child.communicate()[0]
    rc = child.returncode
    return output, rc


def print_progression(percent, width=50, delimiters=['[', ']'], symbol='#'):
    """
    Prints a progress bar to the command line

    Parameters
    ----------
    percent : float
        Completion value between 0 and 100
    width : int, optional
        Number of symbols corresponding to a 100 percent completion
    delimiters : list of str, optional
        Character delimiters for the progression bar
    symbol : str, optional
        Symbol representing one unit of progression
    """
    n_symbols = int(percent/100.0*width)
    progress_bar = delimiters[0] + n_symbols * symbol \
                                 + (width - n_symbols) * ' ' \
                                 + delimiters[1] + " "
    sys.stdout.write("\r" + progress_bar + str(percent) + "%")
    sys.stdout.flush()
