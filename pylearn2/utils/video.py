"""
Utilities for working with videos, pulling out patches, etc.
"""
import numpy

from pylearn2.utils.rng import make_np_rng

__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2011, David Warde-Farley / Universite de Montreal"
__license__ = "BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"
__all__ = ["get_video_dims", "spatiotemporal_cubes"]


def get_video_dims(fname):
    """
    Pull out the frame length, spatial height and spatial width of
    a video file using ffmpeg.

    Parameters
    ----------
    fname : str
        Path to video file to be inspected.

    Returns
    -------
    shape : tuple
        The spatiotemporal dimensions of the video
        (length, height, width).
    """
    try:
        import pyffmpeg
    except ImportError:
        raise ImportError("This function requires pyffmpeg "
                          "<http://code.google.com/p/pyffmpeg/>")
    mp = pyffmpeg.FFMpegReader()
    try:
        mp.open(fname)
        tracks = mp.get_tracks()
        for track in tracks:
            if isinstance(track, pyffmpeg.VideoTrack):
                break
        else:
            raise ValueError('no video track found')
        return (track.duration(),) + track.get_orig_size()
    finally:
        mp.close()


class FrameLookup(object):
    """
    Class encapsulating the logic of turning a frame index into a
    collection of files into the frame index of a specific video file.

    Item-indexing on this object will yield a (filename, nframes, frame_no)
    tuple, where nframes is the number of frames in the given file
    (mainly for checking that we're far enough from the end so that we
    can sample a big enough chunk).

    Parameters
    ----------
    names_ang_lengths : WRITEME
    """
    def __init__(self, names_and_lengths):
        self.files, self.lengths = zip(*names_and_lengths)
        self.terminals = numpy.cumsum([s[1] for s in names_and_lengths])

    def __getitem__(self, i):
        idx = (i < self.terminals).nonzero()[0][0]
        frame_no = i
        if idx > 0:
            frame_no -= self.terminals[idx - 1]
        return self.files[idx], self.lengths[idx], frame_no

    def __len__(self):
        return self.terminals[-1]

    def __iter__(self):
        raise TypeError('iteration not supported')


def spatiotemporal_cubes(file_tuples, shape, n_patches=numpy.inf, rng=None):
    """
    Generator function that yields a stream of (filename, slicetuple)
    representing a spatiotemporal patch of that file.

    Parameters
    ----------
    file_tuples : list of tuples
        Each element should be a 2-tuple consisting of a filename
        (or arbitrary identifier) and a (length, height, width)
        shape tuple of the dimensions (number of frames in the video,
        height and width of each frame).

    shape : tuple
        A shape tuple consisting of the desired (length, height, width)
        of each spatiotemporal patch.

    n_patches : int, optional
        The number of patches to generate. By default, generates patches
        infinitely.

    rng : RandomState object or seed, optional
        The random number generator (or seed) to use. Defaults to None,
        meaning it will be seeded from /dev/urandom or the clock.

    Returns
    -------
    generator : generator object
        A generator that yields a stream of (filename, slicetuple) tuples.
        The slice tuple is such that it indexes into a 3D array containing
        the entire clip with frames indexed along the first axis, rows
        along the second and columns along the third.
    """
    frame_lookup = FrameLookup([(a, b[0]) for a, b in file_tuples])
    file_lookup = dict(file_tuples)
    patch_length, patch_height, patch_width = shape
    done = 0
    rng = make_np_rng(rng, which_method="random_integers")
    while done < n_patches:
        frame = numpy.random.random_integers(0, len(frame_lookup) - 1)
        filename, file_length, frame_no = frame_lookup[frame]
        # Check that there is a contiguous block of frames starting at
        # frame_no that is at least as long as our desired cube length.
        if file_length - frame_no < patch_length:
            continue
        _, video_height, video_width = file_lookup[filename][:3]
        # The last row and column in which a patch could "start" to still
        # fall within frame.
        last_row = video_height - patch_height
        last_col = video_width - patch_width
        row = numpy.random.random_integers(0, last_row)
        col = numpy.random.random_integers(0, last_col)
        patch_slice = (slice(frame_no, frame_no + patch_length),
                       slice(row, row + patch_height),
                       slice(col, col + patch_width))
        done += 1
        yield filename, patch_slice
