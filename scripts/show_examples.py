import sys
from framework.gui import patch_viewer

assert len(sys.argv) == 2
path = sys.argv[1]

if path.endswith('.pkl'):
    from framework.utils import serial
    dataset = serial.load(path)
else:
    print 'not sure what to do with that kind of file'
    quit(-1)

rows = 10
cols = 10

examples = dataset.get_batch_topo(rows*cols)

if len(examples) != 4:
    print 'sorry, view_examples.py only supports image examples for now.'
    print 'this dataset has '+str(len(examples)-2)+' topological dimensions'
    quit(-1)
#

pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3])

for i in xrange(rows*cols):
    pv.add_patch(examples[i,:,:,:])
#

pv.show()
