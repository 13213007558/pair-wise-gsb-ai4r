# frozen_string_literal: true

require_relative '../test_helper'
require 'ai4r/neural_network/gru'

class GRUTest < Minitest::Test
  INPUTS = [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2], [0.2, 0.1]].freeze
  TARGETS = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]].freeze
  H0 = [0.1, -0.2, 0.3].freeze
  EPS = 1e-5
  DELTA = 1e-4

  def setup
    @gru = Ai4r::NeuralNetwork::GRU.new(input_size: 2, hidden_size: 3, output_size: 2, seed: 7)
  end

  # Numerical gradient check on a tiny network, covering the update gate,
  # the reset gate, the candidate state, the output layer and the initial
  # hidden state.
  def test_update_gate_gradients_match_numerical
    assert_parameters_match_numerical(%i[w_z u_z b_z])
  end

  def test_reset_gate_gradients_match_numerical
    assert_parameters_match_numerical(%i[w_r u_r b_r])
  end

  def test_candidate_and_output_gradients_match_numerical
    assert_parameters_match_numerical(%i[w_h u_h b_h w_y b_y])
  end

  def test_initial_state_gradient_matches_numerical
    h0 = H0.dup
    _, grads, = @gru.loss_and_gradients(INPUTS, TARGETS, h0)
    assert_gradient_close(grads[:initial_state], h0) do
      @gru.loss_and_gradients(INPUTS, TARGETS, h0).first
    end
  end

  def test_seeded_initialization_is_reproducible
    a = Ai4r::NeuralNetwork::GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 42)
    b = Ai4r::NeuralNetwork::GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 42)
    c = Ai4r::NeuralNetwork::GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 43)
    assert_equal a.parameters, b.parameters
    refute_equal a.parameters, c.parameters
  end

  def test_eval_returns_one_output_per_step
    out = @gru.eval(INPUTS)
    assert_equal INPUTS.length, out.length
    out.each { |y| assert_equal 2, y.length }
  end

  def test_state_carries_over_between_eval_calls
    seq = [[0.1, 0.0], [0.2, 0.1], [0.0, 0.3], [0.4, 0.2]]
    @gru.reset_state
    full = @gru.eval(seq)
    @gru.reset_state
    first = @gru.eval(seq[0..1])
    second = @gru.eval(seq[2..3])
    assert_equal full[0..1], first
    assert_equal full[2..3], second
  end

  def test_reset_state_restores_initial_behavior
    seq = [[0.1, 0.2], [0.3, 0.4]]
    first = @gru.eval(seq)
    @gru.eval(seq) # advances the state
    @gru.reset_state
    assert_equal first, @gru.eval(seq)
  end

  def test_hidden_state_accessor
    @gru.hidden_state = [0.5, -0.5, 0.25]
    assert_equal [0.5, -0.5, 0.25], @gru.hidden_state
    assert_raises(ArgumentError) { @gru.hidden_state = [1.0] }
  end

  def test_eval_accepts_explicit_initial_state
    seq = [[0.2, 0.1], [0.0, 0.3]]
    @gru.reset_state
    with_argument = @gru.eval(seq, initial_state: [0.1, 0.2, 0.3])
    @gru.hidden_state = [0.1, 0.2, 0.3]
    with_assignment = @gru.eval(seq)
    assert_equal with_argument, with_assignment
  end

  def test_nil_padding_does_not_change_loss_or_state
    seq = [[0.1], [0.2], [0.3]]
    tgt = [[0.2], [0.3], [0.4]]
    gru_plain = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 4, output_size: 1, seed: 9)
    gru_padded = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 4, output_size: 1, seed: 9)
    plain = gru_plain.train([seq], [tgt], epochs: 5, learning_rate: 0.1, truncation: 2)
    padded = gru_padded.train([seq + [nil, nil]], [tgt + [nil, nil]],
                              epochs: 5, learning_rate: 0.1, truncation: 2)
    assert_equal plain, padded
  end

  def test_batch_with_varying_sequence_lengths
    sequences = [[[0.0], [1.0], [0.0], [1.0]], [[1.0], [1.0]]]
    targets = [[[1.0], [0.0], [1.0], [0.0]], [[1.0], [0.0]]]
    gru = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 4, output_size: 1, seed: 2)
    losses = gru.train(sequences, targets, epochs: 4, learning_rate: 0.1, truncation: 3)
    assert_equal 4, losses.length
    assert(losses.all?(&:finite?))
  end

  def test_truncated_bptt_learns_short_dependencies
    rng = Random.new(5)
    sequences = []
    targets = []
    16.times do
      bits = Array.new(6) { rng.rand(2).to_f }
      sequences << bits.map { |b| [b] }
      targets << ([0.0] + bits[0..-2]).map { |b| [b] }
    end
    gru = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 8, output_size: 1, seed: 4)
    losses = gru.train(sequences, targets, epochs: 200, learning_rate: 0.5, truncation: 2)
    assert losses.last < losses.first * 0.8,
           "expected loss to decrease, got #{losses.first} -> #{losses.last}"
  end

  # Delayed copy: the bit shown at the first time step must be reproduced
  # at the last one, four steps later. The truncation window covers the
  # whole sequence so the gradient can reach the first step.
  def test_delayed_copy_task_learns
    rng = Random.new(11)
    sequences = []
    targets = []
    16.times do
      bit = rng.rand(2).to_f
      sequences << [[bit], [0.0], [0.0], [0.0], [0.0]]
      targets << [[0.0], [0.0], [0.0], [0.0], [bit]]
    end
    gru = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 8, output_size: 1, seed: 3)
    losses = gru.train(sequences, targets, epochs: 600, learning_rate: 1.0, truncation: 5)
    assert losses.last < losses.first * 0.2,
           "expected loss to drop, got #{losses.first} -> #{losses.last}"
    correct = sequences.zip(targets).count do |seq, tgt|
      gru.reset_state
      (gru.eval(seq).last[0] - tgt.last[0]).abs < 0.5
    end
    assert_equal sequences.size, correct
  end

  private

  def assert_parameters_match_numerical(keys)
    _, grads, = @gru.loss_and_gradients(INPUTS, TARGETS, H0)
    keys.each do |key|
      assert_gradient_close(grads[key], @gru.parameters[key]) do
        @gru.loss_and_gradients(INPUTS, TARGETS, H0).first
      end
    end
  end

  def assert_gradient_close(analytic, container)
    each_leaf_path(container) do |path|
      original = fetch(container, path)
      store(container, path, original + EPS)
      plus = yield
      store(container, path, original - EPS)
      minus = yield
      store(container, path, original)
      numerical = (plus - minus) / (2.0 * EPS)
      assert_in_delta fetch(analytic, path), numerical, DELTA,
                      "gradient mismatch at #{path.inspect}"
    end
  end

  def each_leaf_path(container, path = [], &block)
    container.each_with_index do |value, i|
      if value.is_a?(Array)
        each_leaf_path(value, path + [i], &block)
      else
        yield(path + [i])
      end
    end
  end

  def fetch(container, path)
    path.reduce(container) { |c, i| c[i] }
  end

  def store(container, path, value)
    parent = path[0..-2].reduce(container) { |c, i| c[i] }
    parent[path[-1]] = value
  end
end
