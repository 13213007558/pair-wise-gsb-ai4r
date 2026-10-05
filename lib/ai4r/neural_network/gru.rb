# frozen_string_literal: true

# Minimal GRU (Gated Recurrent Unit) implementation.
# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
# Url::       https://github.com/SergioFierens/ai4r

module Ai4r
  module NeuralNetwork
    # = Introduction
    #
    # A small GRU (Gated Recurrent Unit) network for short sequences, in the
    # spirit of Cho et al. (2014). The output at time t depends not only on
    # the current input but also on a hidden state summarizing previous
    # observations. Pure Ruby, no tensor framework required.
    #
    # = Features
    #
    # * Forward inference over sequences (eval / step)
    # * Training with truncated backpropagation through time (BPTT)
    # * Explicit hidden state management: hidden_state, set_hidden_state,
    #   reset_state
    # * Masked (padded) steps: nil inputs never update the hidden state and
    #   nil targets never contribute to the loss
    # * Reproducible weight initialization via :seed
    #
    # = Usage
    #
    #   net = Ai4r::NeuralNetwork::GRU.new(input_size: 2, hidden_size: 8,
    #                                      output_size: 1, seed: 42)
    #   net.reset_state
    #   outputs = net.eval([[0.1, 0.2], [0.3, 0.4]])
    #   net.train([[[0.1, 0.2], [0.3, 0.4]]], [[[0.5], [0.6]]],
    #             epochs: 100, learning_rate: 0.1, truncation: 4)
    class GRU
      # Weight matrices, keyed by name. Gate weights are stored as
      # hidden_size x fan_in matrices (row per hidden unit); :w_y is
      # output_size x hidden_size. Biases are plain vectors.
      attr_reader :weights

      attr_reader :input_size, :hidden_size, :output_size

      # input_size::  size of each input vector x_t
      # hidden_size:: size of the hidden state
      # output_size:: size of each output vector y_t (linear layer)
      # seed::        optional seed for reproducible weight initialization
      def initialize(input_size:, hidden_size:, output_size:, seed: nil)
        @input_size = input_size
        @hidden_size = hidden_size
        @output_size = output_size
        @rng = seed ? Random.new(seed) : Random.new
        init_weights
        reset_state
      end

      # Run a whole sequence of input vectors through the network, starting
      # from the current hidden state. Returns one output vector per step.
      # A nil input marks a padded step: the hidden state is left untouched
      # and the corresponding output is nil.
      def eval(inputs)
        inputs.map { |x| x.nil? ? nil : step(x) }
      end

      # Run a single time step and return the output vector. Updates the
      # current hidden state.
      def step(x)
        check_vector!(x, @input_size, 'input')
        h, y, = forward_step(x, @hidden_state)
        @hidden_state = h
        y
      end

      # Current hidden state (a copy; mutate the network via set_hidden_state).
      def hidden_state
        @hidden_state.dup
      end

      # Explicitly set the hidden state, e.g. to continue a sequence from a
      # state previously returned by hidden_state or train's :final_states.
      def set_hidden_state(state)
        check_vector!(state, @hidden_size, 'hidden state')
        @hidden_state = state.map(&:to_f)
      end

      # Reset the hidden state to zeros. Call this to explicitly restart a
      # sequence.
      def reset_state
        @hidden_state = Array.new(@hidden_size, 0.0)
      end

      # Train on a batch of sequences with truncated BPTT.
      #
      # sequences:: array of sequences; each sequence is an array of input
      #             vectors. Sequences may have different lengths; a nil
      #             entry marks a padded step (no state update, no loss).
      # targets::   parallel array of target vectors; a nil entry means the
      #             step contributes no loss.
      # epochs::    number of passes over the batch
      # learning_rate:: SGD step size
      # truncation::  chunk length for truncated BPTT. Hidden states carry
      #             over from one chunk to the next, but gradients never
      #             flow across a chunk boundary. nil means full BPTT.
      # initial_states:: optional array (one entry per sequence) of hidden
      #             states to start from at the beginning of every epoch;
      #             nil entries (or a nil array) reset that sequence to a
      #             zero state. Use this to continue sequences across train
      #             calls, or to explicitly reset individual sequences.
      #
      # Returns { losses: [mean loss per epoch], final_states: [...] }.
      def train(sequences, targets, epochs: 1, learning_rate: 0.1, truncation: nil, initial_states: nil)
        raise ArgumentError, 'sequences and targets must have the same size' if sequences.length != targets.length

        start_states = Array.new(sequences.length) { |i| initial_states && initial_states[i] ? initial_states[i].dup : nil }
        states = start_states.map { |s| s&.dup }
        losses = []
        epochs.times do
          total_loss = 0.0
          total_steps = 0
          sequences.each_with_index do |seq, si|
            h = start_states[si] ? start_states[si].dup : Array.new(@hidden_size, 0.0)
            each_chunk(seq.length, truncation) do |first, last|
              result = loss_and_gradients(seq[first..last], targets[si][first..last], initial_state: h)
              apply_gradients(result[:gradients], learning_rate)
              total_loss += result[:loss] * result[:labeled_steps]
              total_steps += result[:labeled_steps]
              h = result[:final_state]
            end
            states[si] = h
          end
          losses << (total_steps.positive? ? total_loss / total_steps : 0.0)
        end
        { losses: losses, final_states: states }
      end

      # Forward and backward (full BPTT) over a single sequence chunk.
      # The initial_state is treated as a constant: its gradient is reported
      # as :state_gradient but never applied, which is exactly what makes
      # chunk boundaries truncating when train chains this method.
      #
      # Returns a hash with:
      #   loss::            mean squared error over labeled steps
      #   gradients::       hash of weight gradients (same keys as #weights)
      #   state_gradient::  gradient of the loss w.r.t. initial_state
      #   final_state::     hidden state after the last step
      #   labeled_steps::   number of steps that contributed to the loss
      def loss_and_gradients(inputs, targets, initial_state: nil)
        raise ArgumentError, 'inputs and targets must have the same length' if inputs.length != targets.length

        h = initial_state ? initial_state.dup : Array.new(@hidden_size, 0.0)
        caches = []
        loss = 0.0
        labeled = 0
        inputs.each_with_index do |x, t|
          if x.nil?
            caches << nil
            next
          end
          check_vector!(x, @input_size, 'input')
          h_prev = h
          h, y, cache = forward_step(x, h_prev)
          caches << cache
          next if targets[t].nil?

          check_vector!(targets[t], @output_size, 'target')
          labeled += 1
          y.each_index { |i| loss += 0.5 * (y[i] - targets[t][i])**2 }
        end
        loss /= labeled if labeled.positive?

        grads = zero_gradients
        dh_next = Array.new(@hidden_size, 0.0)
        (inputs.length - 1).downto(0) do |t|
          cache = caches[t]
          next if cache.nil?

          dh = dh_next
          unless targets[t].nil?
            dy = cache[:y].each_index.map { |i| (cache[:y][i] - targets[t][i]) / (labeled.positive? ? labeled : 1) }
            dh = vec_add(dh, backward_output(dy, cache[:h], grads))
          end
          dh_next = backward_step(dh, cache, grads)
        end

        { loss: loss, gradients: grads, state_gradient: dh_next, final_state: h, labeled_steps: labeled }
      end

      private

      def init_weights
        @weights = {
          w_zx: random_matrix(@hidden_size, @input_size),
          w_zh: random_matrix(@hidden_size, @hidden_size),
          b_z: Array.new(@hidden_size, 0.0),
          w_rx: random_matrix(@hidden_size, @input_size),
          w_rh: random_matrix(@hidden_size, @hidden_size),
          b_r: Array.new(@hidden_size, 0.0),
          w_nx: random_matrix(@hidden_size, @input_size),
          w_nh: random_matrix(@hidden_size, @hidden_size),
          b_n: Array.new(@hidden_size, 0.0),
          w_y: random_matrix(@output_size, @hidden_size),
          b_y: Array.new(@output_size, 0.0)
        }
      end

      def random_matrix(rows, cols)
        limit = Math.sqrt(6.0 / (rows + cols))
        Array.new(rows) { Array.new(cols) { (@rng.rand * 2 * limit) - limit } }
      end

      # One GRU step:
      #   z = sigmoid(W_zx x + W_zh h' + b_z)        (update gate)
      #   r = sigmoid(W_rx x + W_rh h' + b_r)        (reset gate)
      #   n = tanh(W_nx x + W_nh (r * h') + b_n)     (candidate)
      #   h = (1 - z) * n + z * h'
      #   y = W_y h + b_y                            (linear output)
      def forward_step(x, h_prev)
        z = vec_add(mat_vec(@weights[:w_zx], x), mat_vec(@weights[:w_zh], h_prev), @weights[:b_z]).map { |v| sigmoid(v) }
        r = vec_add(mat_vec(@weights[:w_rx], x), mat_vec(@weights[:w_rh], h_prev), @weights[:b_r]).map { |v| sigmoid(v) }
        rh = hadamard(r, h_prev)
        n = vec_add(mat_vec(@weights[:w_nx], x), mat_vec(@weights[:w_nh], rh), @weights[:b_n]).map { |v| Math.tanh(v) }
        h = Array.new(@hidden_size) { |i| ((1.0 - z[i]) * n[i]) + (z[i] * h_prev[i]) }
        y = vec_add(mat_vec(@weights[:w_y], h), @weights[:b_y])
        cache = { x: x, h_prev: h_prev, z: z, r: r, rh: rh, n: n, h: h, y: y }
        [h, y, cache]
      end

      # Gradient of the output layer. Accumulates into grads and returns the
      # gradient w.r.t. the hidden state h.
      def backward_output(dy, h, grads)
        dy.each_with_index do |d, i|
          @hidden_size.times { |j| grads[:w_y][i][j] += d * h[j] }
          grads[:b_y][i] += d
        end
        mat_t_vec(@weights[:w_y], dy)
      end

      # Gradient through one GRU cell, given dh (gradient w.r.t. h_t).
      # Accumulates weight gradients into grads and returns the gradient
      # w.r.t. h_{t-1}.
      def backward_step(dh, cache, grads)
        x = cache[:x]
        h_prev = cache[:h_prev]
        z = cache[:z]
        r = cache[:r]
        rh = cache[:rh]
        n = cache[:n]

        dz = Array.new(@hidden_size) { |i| dh[i] * (h_prev[i] - n[i]) }
        dn = Array.new(@hidden_size) { |i| dh[i] * (1.0 - z[i]) }

        dan = Array.new(@hidden_size) { |i| dn[i] * (1.0 - n[i]**2) }
        daz = Array.new(@hidden_size) { |i| dz[i] * z[i] * (1.0 - z[i]) }

        drh = mat_t_vec(@weights[:w_nh], dan)
        dr = Array.new(@hidden_size) { |i| drh[i] * h_prev[i] }
        dar = Array.new(@hidden_size) { |i| dr[i] * r[i] * (1.0 - r[i]) }

        accumulate(grads[:w_zx], daz, x)
        accumulate(grads[:w_zh], daz, h_prev)
        vec_accumulate(grads[:b_z], daz)
        accumulate(grads[:w_rx], dar, x)
        accumulate(grads[:w_rh], dar, h_prev)
        vec_accumulate(grads[:b_r], dar)
        accumulate(grads[:w_nx], dan, x)
        accumulate(grads[:w_nh], dan, rh)
        vec_accumulate(grads[:b_n], dan)

        dh_prev = Array.new(@hidden_size) { |i| (dh[i] * z[i]) + (drh[i] * r[i]) }
        vec_accumulate(dh_prev, mat_t_vec(@weights[:w_zh], daz))
        vec_accumulate(dh_prev, mat_t_vec(@weights[:w_rh], dar))
        dh_prev
      end

      def zero_gradients
        {
          w_zx: zero_matrix(@hidden_size, @input_size),
          w_zh: zero_matrix(@hidden_size, @hidden_size),
          b_z: Array.new(@hidden_size, 0.0),
          w_rx: zero_matrix(@hidden_size, @input_size),
          w_rh: zero_matrix(@hidden_size, @hidden_size),
          b_r: Array.new(@hidden_size, 0.0),
          w_nx: zero_matrix(@hidden_size, @input_size),
          w_nh: zero_matrix(@hidden_size, @hidden_size),
          b_n: Array.new(@hidden_size, 0.0),
          w_y: zero_matrix(@output_size, @hidden_size),
          b_y: Array.new(@output_size, 0.0)
        }
      end

      def apply_gradients(grads, learning_rate)
        @weights.each do |name, w|
          g = grads[name]
          if w.first.is_a?(Array)
            w.each_with_index do |row, i|
              row.each_index { |j| row[j] -= learning_rate * g[i][j] }
            end
          else
            w.each_index { |i| w[i] -= learning_rate * g[i] }
          end
        end
      end

      def each_chunk(length, truncation)
        step = truncation&.positive? ? truncation : length
        first = 0
        while first < length
          last = [first + step, length].min - 1
          yield first, last
          first = last + 1
        end
      end

      def check_vector!(vec, size, name)
        raise ArgumentError, "#{name} must be an Array of size #{size}" unless vec.is_a?(Array) && vec.length == size
      end

      def sigmoid(x)
        1.0 / (1.0 + Math.exp(-x))
      end

      def mat_vec(mat, vec)
        mat.map do |row|
          sum = 0.0
          row.each_index { |j| sum += row[j] * vec[j] }
          sum
        end
      end

      def mat_t_vec(mat, vec)
        cols = mat.first.length
        Array.new(cols) do |j|
          sum = 0.0
          mat.each_index { |i| sum += mat[i][j] * vec[i] }
          sum
        end
      end

      def vec_add(*vecs)
        vecs.inject { |acc, v| acc.each_index.map { |i| acc[i] + v[i] } }
      end

      def vec_accumulate(acc, vec)
        acc.each_index { |i| acc[i] += vec[i] }
        acc
      end

      def hadamard(a, b)
        a.each_index.map { |i| a[i] * b[i] }
      end

      def accumulate(mat, d_row, vec)
        d_row.each_with_index do |d, i|
          row = mat[i]
          vec.each_index { |j| row[j] += d * vec[j] }
        end
      end

      def zero_matrix(rows, cols)
        Array.new(rows) { Array.new(cols, 0.0) }
      end
    end
  end
end
