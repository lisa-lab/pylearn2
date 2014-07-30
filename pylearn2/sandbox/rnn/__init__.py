"""
The RNN framework. It extends the standard MLP framework in
several ways, most notably:

- Wrap the set_input_space, get_output_space and fprop methods to
  deal with sequential data. It does so by reshaping the data from
  a ND [time, batch, data, ..., data] tensor to a (N-1)D tensor,
  [time * batch, data, ..., data] and reshaping the data back before
  passing the output to the next layer.

- Recurrent layers are introduced.

- A SequenceSpace is defined, which can be a sequence of any other space
  e.g. SequenceSpace(VectorSpace(dim=100)).

- The dataset iterators are adapted to deal with sequential data, either
  by creating batches which are uniform in sequence length, or by
  providing a mask along with zero-padded data, describing the length of
  the sequences.
"""
