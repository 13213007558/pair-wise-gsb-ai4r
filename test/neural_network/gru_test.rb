# frozen_string_literal: true

#
# gru_test.rb
#
# Unit tests for the GRU (gated recurrent unit) network.
#
# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
# Url::       https://github.com/SergioFierens/ai4r
#

require 'ai4r/neural_network/gru'
require_relative '../test_helper'

module Ai4r
  module NeuralNetwork
    class GRUTest < Minitest::Test
      WEIGHT_KEYS = %i[w_zx w_zh b_z w_rx w_rh b_r w_nx w_nh b_n w_y b_y].freeze

      def build_net(seed: 42)
        GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: seed)
      end

      def test_eval_returns_one_output_per_step
        net = build_net
        outputs = net.eval([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        assert_equal 3, outputs.length
        outputs.each { |y| assert_equal 1, y.length }
      end

      def test_eval_rejects_wrong_input_size
        net = build_net
        assert_raises(ArgumentError) { net.eval([[1.0, 2.0, 3.0]]) }
      end

      def test_hidden_state_management
        net = build_net
        assert_equal [0.0, 0.0, 0.0], net.hidden_state
        net.eval([[0.3, -0.2]])
        carried = net.hidden_state
        refute_equal [0.0, 0.0, 0.0], carried

        # State carries over between eval calls.
        net.eval([[0.1, 0.1]])
        refute_equal carried, net.hidden_state

        # Explicitly handing a state back restores it.
        net.set_hidden_state(carried)
        assert_equal carried, net.hidden_state

        # Explicit reset restarts the sequence.
        net.reset_state
        assert_equal [0.0, 0.0, 0.0], net.hidden_state

        assert_raises(ArgumentError) { net.set_hidden_state([1.0, 2.0]) }
      end

      def test_reproducible_initialization
        a = GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 7)
        b = GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 7)
        c = GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 8)
        WEIGHT_KEYS.each do |key|
          assert_equal a.weights[key], b.weights[key]
          refute_equal a.weights[key], c.weights[key] unless key.to_s.start_with?('b_')
        end
      end

      def test_nil_inputs_do_not_update_state
        net = build_net
        net.eval([[0.4, 0.1]])
        state_after_first = net.hidden_state
        net.eval([nil, nil])
        assert_equal state_after_first, net.hidden_state

        net.reset_state
        outputs = net.eval([[0.4, 0.1], nil, [0.2, 0.3]])
        assert_nil outputs[1]
        refute_nil outputs[2]
      end

      def test_padded_steps_are_invisible_to_loss_and_gradients
        net = build_net
        xs = [[0.3, -0.1], [0.2, 0.4]]
        ts = [[0.1], [0.2]]
        plain = net.loss_and_gradients(xs, ts)
        padded = net.loss_and_gradients([xs[0], nil, xs[1], nil], [ts[0], nil, ts[1], nil])
        assert_in_delta plain[:loss], padded[:loss], 1e-12
        WEIGHT_KEYS.each do |key|
          assert_approximate_equality_of_nested_list plain[:gradients][key], padded[:gradients][key], 1e-12
        end
        assert_approximate_equality_of_nested_list plain[:final_state], padded[:final_state], 1e-12
      end

      def test_nil_targets_contribute_no_loss
        net = build_net
        xs = [[0.3, -0.1], [0.2, 0.4]]
        masked = net.loss_and_gradients(xs, [[0.1], nil])
        single = net.loss_and_gradients([xs[0]], [[0.1]])
        assert_in_delta single[:loss], masked[:loss], 1e-12
        assert_equal 1, masked[:labeled_steps]
      end

      def test_numerical_gradient_check
        net = GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 99)
        inputs = [[0.4, -0.2], [0.1, 0.3], [-0.3, 0.2], [0.2, 0.1]]
        targets = [[0.2], [-0.1], [0.3], [0.0]]
        h0 = [0.1, -0.2, 0.05]
        analytic = net.loss_and_gradients(inputs, targets, initial_state: h0)

        WEIGHT_KEYS.each do |key|
          numerical = numerical_weight_gradient(net, inputs, targets, h0, key)
          assert_approximate_equality_of_nested_list analytic[:gradients][key], numerical, 1e-4
        end

        numerical_state = numerical_state_gradient(net, inputs, targets, h0)
        assert_approximate_equality_of_nested_list analytic[:state_gradient], numerical_state, 1e-4
      end

      def test_truncation_carries_state_but_cuts_gradient
        net = GRU.new(input_size: 2, hidden_size: 3, output_size: 1, seed: 5)
        xs = [[0.3, -0.1], [0.2, 0.4], [-0.2, 0.1], [0.5, 0.0]]
        ts = [[0.1], [0.2], [0.0], [0.3]]
        zeros = [0.0, 0.0, 0.0]

        chunk1 = net.loss_and_gradients(xs[0..1], ts[0..1], initial_state: zeros)
        chunk2 = net.loss_and_gradients(xs[2..3], ts[2..3], initial_state: chunk1[:final_state])
        full = net.loss_and_gradients(xs, ts, initial_state: zeros)

        # Full BPTT lets the loss flow back into the initial state; chunk 2
        # would too, but a truncation boundary must drop that gradient.
        refute_all_zero full[:state_gradient]
        refute_all_zero chunk2[:state_gradient]
        refute_nested_close full[:gradients][:w_zh], add_nested(chunk1[:gradients][:w_zh], chunk2[:gradients][:w_zh])

        # One epoch with truncation 2 must apply exactly the sum of the two
        # chunk gradients: state is carried forward, nothing flows back.
        before = deep_copy(net.weights)
        net.train([xs], [ts], epochs: 1, learning_rate: 0.5, truncation: 2)
        WEIGHT_KEYS.each do |key|
          expected = sub_nested(before[key], scale_nested(add_nested(chunk1[:gradients][key], chunk2[:gradients][key]), 0.5))
          assert_approximate_equality_of_nested_list expected, net.weights[key], 1e-12
        end
      end

      def test_variable_length_batch_and_explicit_reset
        net = GRU.new(input_size: 1, hidden_size: 4, output_size: 1, seed: 3)
        seqs = [[[0.1], [0.2], [0.3]], [[0.5], [0.4], [0.3], [0.2], [0.1]]]
        tgts = [[[0.2], [0.3], [0.4]], [[0.4], [0.3], [0.2], [0.1], [0.0]]]

        result = net.train(seqs, tgts, epochs: 3, learning_rate: 0.05, truncation: 2)
        assert_equal 3, result[:losses].length
        assert(result[:losses].all?(&:finite?))
        assert_equal [4, 4], result[:final_states].map(&:length)

        # nil initial state resets that sequence; omitted states reset all.
        fresh = GRU.new(input_size: 1, hidden_size: 4, output_size: 1, seed: 3)
        reset_run = fresh.train(seqs, tgts, epochs: 1, learning_rate: 0.05)

        twin = GRU.new(input_size: 1, hidden_size: 4, output_size: 1, seed: 3)
        zeros = [Array.new(4, 0.0), Array.new(4, 0.0)]
        explicit_reset = twin.train(seqs, tgts, epochs: 1, learning_rate: 0.05, initial_states: zeros)
        assert_equal reset_run[:losses], explicit_reset[:losses]

        # Handing the final states back continues the sequences instead of
        # resetting them, which changes the trajectory.
        continued = twin.train(seqs, tgts, epochs: 1, learning_rate: 0.05,
                                           initial_states: reset_run[:final_states])
        refute_equal reset_run[:losses], continued[:losses]
      end

      def test_delayed_copy_task_learns
        delay = 2
        length = 12
        rng = Random.new(7)
        sequences = Array.new(6) { Array.new(length) { [rng.rand.round.to_f] } }
        targets = sequences.map do |seq|
          seq.each_with_index.map { |_, t| t < delay ? [0.0] : seq[t - delay].dup }
        end

        net = GRU.new(input_size: 1, hidden_size: 8, output_size: 1, seed: 42)
        result = net.train(sequences, targets, epochs: 300, learning_rate: 0.1, truncation: 4)
        assert result[:losses].last < result[:losses].first, 'loss should decrease'
        assert result[:losses].last < 0.02, "final loss too high: #{result[:losses].last}"

        net.reset_state
        predictions = net.eval(sequences.first).map { |y| y[0] }
        targets.first.each_with_index do |target, t|
          assert_in_delta target[0], predictions[t], 0.3, "step #{t}: expected #{target[0]}, got #{predictions[t]}"
        end
      end

      private

      def numerical_weight_gradient(net, inputs, targets, h0, key, eps = 1e-5)
        w = net.weights[key]
        if w.first.is_a?(Array)
          w.each_with_index.map do |row, _i|
            row.each_index.map { |j| central_difference(net, inputs, targets, h0, eps) { |d| row[j] += d } }
          end
        else
          w.each_index.map { |i| central_difference(net, inputs, targets, h0, eps) { |d| w[i] += d } }
        end
      end

      def numerical_state_gradient(net, inputs, targets, h0, eps = 1e-5)
        h0.each_index.map do |k|
          central_difference(net, inputs, targets, h0, eps) { |d| h0[k] += d }
        end
      end

      def central_difference(net, inputs, targets, h0, eps)
        yield eps
        plus = net.loss_and_gradients(inputs, targets, initial_state: h0)[:loss]
        yield(-2 * eps)
        minus = net.loss_and_gradients(inputs, targets, initial_state: h0)[:loss]
        yield eps
        (plus - minus) / (2 * eps)
      end

      def refute_all_zero(vec)
        assert vec.any? { |v| v.abs > 1e-8 }, "expected a non-zero gradient, got #{vec.inspect}"
      end

      def refute_nested_close(a, b, tol = 1e-6)
        flat_a = a.flatten
        flat_b = b.flatten
        max_diff = flat_a.zip(flat_b).map { |x, y| (x - y).abs }.max
        assert max_diff > tol, "expected gradients to differ, max diff #{max_diff}"
      end

      def deep_copy(obj)
        obj.map { |k, v| [k, v.first.is_a?(Array) ? v.map(&:dup) : v.dup] }.to_h
      end

      def add_nested(a, b)
        a.zip(b).map do |x, y|
          x.is_a?(Array) ? add_nested(x, y) : x + y
        end
      end

      def sub_nested(a, b)
        a.zip(b).map do |x, y|
          x.is_a?(Array) ? sub_nested(x, y) : x - y
        end
      end

      def scale_nested(a, factor)
        a.map { |x| x.is_a?(Array) ? scale_nested(x, factor) : x * factor }
      end
    end
  end
end
