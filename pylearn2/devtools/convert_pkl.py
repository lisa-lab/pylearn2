"""
Convert a .pkl from Python 2 to Python 3
"""

import pickle
import sys


if __name__ == "__main__":
    source_pkl = sys.argv[1]
    try:
        target_pkl = sys.argv[2]
    except IndexError:
        target_pkl = source_pkl
    with open(source_pkl, 'rb') as f:
        obj = pickle.load(f, encoding='latin-1')
    with open(target_pkl, 'wb') as f:
        pickle.dump(obj, f)
