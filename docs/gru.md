# GRU — Gated Recurrent Unit

`Ai4r::NeuralNetwork::GRU` is a small recurrent network for short sequences.
The output at time *t* depends not only on the current input but also on a
hidden state that summarizes the previous observations, which makes it suitable
for tasks where context matters (e.g. delayed echoes of earlier inputs). It is
implemented in pure Ruby — no tensor framework required.

## Features

* Forward inference over sequences with `eval` / `step`.
* Training with truncated backpropagation through time (BPTT).
* Explicit hidden state management: `hidden_state`, `set_hidden_state` and
  `reset_state`.
* Masked (padded) steps: a `nil` input never updates the hidden state and a
  `nil` target never contributes to the loss, so batches of variable-length
  sequences are safe.
* Reproducible weight initialization via the `seed` parameter.

## Usage

```ruby
require 'ai4r/neural_network/gru'

net = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 8, output_size: 1, seed: 42)

# Forward inference; the hidden state carries over between calls.
net.reset_state
outputs = net.eval([[0.0], [1.0], [0.0]])

# Training a batch of sequences (they may have different lengths).
sequences = [[[0.0], [1.0], [1.0]], [[1.0], [0.0]]]
targets   = [[[0.0], [0.0], [1.0]], [[0.0], [1.0]]]
result = net.train(sequences, targets, epochs: 100, learning_rate: 0.1, truncation: 4)
result[:losses]        # mean loss per epoch
result[:final_states]  # hidden state at the end of each sequence
```

## Truncated BPTT and state handling

`truncation: k` splits each sequence into chunks of `k` steps. The hidden
state **carries over** from one chunk to the next, but gradients never flow
across a chunk boundary — this is the classic truncated BPTT. Omit the
parameter (or pass `nil`) for full backpropagation through the whole
sequence.

Each epoch restarts every sequence from `initial_states` (zeros by default).
To continue sequences across `train` calls, feed the previous
`final_states` back in; to explicitly reset one sequence, pass `nil` (or a
zero vector) at its position:

```ruby
first  = net.train(sequences, targets, epochs: 1)
second = net.train(sequences, targets, epochs: 1, initial_states: first[:final_states])
reset  = net.train(sequences, targets, epochs: 1, initial_states: [nil, first[:final_states][1]])
```

The same idea applies to inference: `eval` continues from the current hidden
state, `hidden_state` returns a copy of it, `set_hidden_state` installs one,
and `reset_state` zeroes it.

## Gradient checking

`loss_and_gradients(inputs, targets, initial_state: ...)` runs full BPTT over
a single chunk and reports the loss, the weight gradients, and the gradient
with respect to the initial state. The test suite uses it to verify the
analytical gradients of the update gate, the reset gate and the initial state
against finite differences.

## Notes

The implementation is intentionally minimal and is not tuned for convergence
on arbitrary datasets — it is meant for experimentation and teaching.

