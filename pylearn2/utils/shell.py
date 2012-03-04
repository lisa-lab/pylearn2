import subprocess as sp

def run_shell_command(cmd):
    """ Runs cmd as a shell command.
    Waits for it to finish executing, then returns all output
    printed to standard error and standard out, and the return code
    """
    child = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.STDOUT)
    output = child.communicate()[0]
    rc = child.returncode()
    return output, rc
