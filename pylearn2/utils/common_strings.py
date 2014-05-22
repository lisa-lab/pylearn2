environment_variable_essay = """
Platform-specific instructions for setting environment variables:

Linux
=====
On most linux setups, you can define your environment variable by adding this
line to your ~/.bashrc file:

export PYLEARN2_VIEWER_COMMAND="eog --new-instance"

*** YOU MUST INCLUDE THE WORD "export". DO NOT JUST ASSIGN TO THE ENVIRONMENT VARIABLE ***
If you do not include the word "export", the environment variable will be set
in your bash shell, but will not be visible to processes that you launch from
it, like the python interpreter.

Don't forget that changes from your .bashrc file won't apply until you run

source ~/.bashrc

or open a new terminal window. If you're seeing this from an ipython notebook
you'll need to restart the ipython notebook, or maybe modify os.environ from
an ipython cell.

Mac OS X
========

Environment variables on Mac OS X work the same as in Linux, except you should
modify and run the "source" command on ~/.profile rather than ~/.bashrc.
"""
