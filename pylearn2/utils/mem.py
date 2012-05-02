import subprocess
import os

def get_memory_usage():
    """Return int containing memory used by this process.
    Don't trust this too much, I'm not totally sure what ps rss measures
    """

    pid = os.getpid()
    process = subprocess.Popen("ps -o rss %s | awk '{sum+=$1} END {print sum}'" % pid,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    )
    stdout_list = process.communicate()[0].split('\n')
    return int(stdout_list[0])

