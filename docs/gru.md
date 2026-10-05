# GRU — Gated Recurrent Unit Network

A small, pure-Ruby GRU network for short sequences where the output at
each time step depends on the current input *and* on previous
observations. No tensor frameworks, no external dependencies — just
readable matrix math on plain Ruby arrays.

## When to use it

* Sequence labeling and short-term memory tasks (e.g. delayed copy).
* Demos and teaching: the whole forward/backward pass fits in one file.
* Anywhere you need an explicit, inspectable hidden state.

## Quick start

```ruby
require 'ai4r/neural_network/gru'

gru = Ai4r::NeuralNetwork::GRU.new(
  input_size: 1,     # features per time step
  hidden_size: 8,    # recurrent state size
  output_size: 1,    # outputs per time step
  seed: 42           # reproducible weight initialization
)

# Delayed copy: repeat the first bit at the last step.
sequences = [[[1.0], [0.0], [0.0], [0.0], [0.0]]]
targets   = [[[0.0], [0.0], [0.0], [0.0], [1.0]]]
losses = gru.train(sequences, targets, epochs: 100, learning_rate: 0.5, truncation: 5)

gru.reset_state
outputs = gru.eval([[1.0], [0.0], [0.0], [0.0], [0.0]])
```

## Hidden state management

The hidden state carries over between `eval` calls, so consecutive
chunks of a long sequence can be processed one after another:

```ruby
gru.reset_state                 # start a fresh sequence (zero state)
gru.eval(first_chunk)           # state advances
gru.eval(second_chunk)          # continues where the first chunk ended

gru.hidden_state                # => copy of the current state
gru.hidden_state = saved_state  # restore a previously captured state
gru.eval(chunk, initial_state: saved_state) # or pass it per call
```

## Truncated backpropagation through time

`train` splits every sequence into chunks of `truncation` steps. The
hidden state is carried forward across chunks, but gradients never flow
across a chunk boundary — the gradient with respect to the chunk's
initial state is deliberately discarded. Choose a truncation window that
covers the longest dependency you expect the network to learn.

## Batches, padding and resets

* `train` accepts a batch of sequences with **different lengths**.
* Pad sequences with `nil` entries: padded positions neither update the
  hidden state nor contribute to the loss.
* Every epoch starts each sequence from a zero hidden state.
* `seed:` makes weight initialization reproducible.

## Testing hooks

`loss_and_gradients(inputs, targets, initial_state)` returns
`[loss, gradients, final_state]` for a single contiguous chunk. The
gradient hash covers all parameters plus `:initial_state`, which makes
numerical gradient checking straightforward (see
`test/neural_network/gru_test.rb`).

## Limitations

* Educational implementation: no GPU, no batching across sequences in
  the forward pass, no gradient clipping.
* Truncated BPTT cannot learn dependencies longer than the truncation
  window.
* No convergence guarantees on arbitrary datasets.
