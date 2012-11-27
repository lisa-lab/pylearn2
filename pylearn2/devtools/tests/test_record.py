from pylearn2.devtools.record import Record
import cStringIO

def test_record_good():

    """
    Tests that when we record a sequence of events, then
    repeat it exactly, the Record class:
        1) Records it correctly
        2) Does not raise any errors
    """

    # Record a sequence of events
    output = cStringIO.StringIO()

    recorder = Record(file_object=output, replay=False)

    num_lines = 10

    for i in xrange(num_lines):
        recorder.handle_line(str(i)+'\n')

    # Make sure they were recorded correctly
    output_value = output.getvalue()

    assert output_value == ''.join(str(i)+'\n' for i in xrange(num_lines))

    # Make sure that the playback functionality doesn't raise any errors
    # when we repeat them
    output = cStringIO.StringIO(output_value)

    playback_checker = Record(file_object=output,  replay=True)

    for i in xrange(num_lines):
        playback_checker.handle_line(str(i)+'\n')


