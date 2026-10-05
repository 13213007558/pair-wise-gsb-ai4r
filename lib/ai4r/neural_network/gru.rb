# frozen_string_literal: true

# Minimal GRU (Gated Recurrent Unit) network with truncated BPTT.
# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
# Url::       https://github.com/SergioFierens/ai4r

require_relative '../data/parameterizable'

module Ai4r
  module NeuralNetwork
    # A small GRU network implemented in pure Ruby (no tensor frameworks).
    #
    # The network processes sequences of input vectors and produces one
    # output vector per time step. The hidden state can be read, replaced
    # and reset explicitly, so consecutive chunks of a sequence can be
    # evaluated while carrying the state over.
    #
    # Training uses truncated backpropagation through time: each sequence
    # is split into chunks of +truncation+ steps, the hidden state is
    # carried forward across chunks, but gradients are not propagated
    # beyond a chunk boundary. Sequences in a batch may have different
    # lengths; pad them with +nil+ entries, which neither update the
    # hidden state nor contribute to the loss.
    #
    #   gru = Ai4r::NeuralNetwork::GRU.new(input_size: 1, hidden_size: 8,
    #                                      output_size: 1, seed: 42)
    #   losses = gru.train(sequences, targets, epochs: 100, truncation: 8)
    #   gru.reset_state
    #   outputs = gru.eval([[0.0], [1.0], [0.0]])
    class GRU
      include Ai4r::Data::Parameterizable

      parameters_info input_size: 'Number of input features per time step.',
                      hidden_size: 'Number of hidden units.',
                      output_size: 'Number of output features per time step.',
                      output_activation: 'Output activation (:sigmoid, :tanh or :linear).',
                      seed: 'Deterministic random seed for weight initialization.'

      attr_reader :input_size, :hidden_size, :output_size, :output_activation, :seed

      # Trainable parameters, as a hash of arrays (live references):
      # :w_z, :u_z, :b_z (update gate), :w_r, :u_r, :b_r (reset gate),
      # :w_h, :u_h, :b_h (candidate state), :w_y, :b_y (output layer).
      attr_reader :parameters

      PARAMETER_KEYS = %i[w_z u_z b_z w_r u_r b_r w_h u_h b_h w_y b_y].freeze

      def initialize(input_size:, hidden_size:, output_size:,
                     output_activation: :sigmoid, seed: nil)
        raise ArgumentError, 'output_activation must be :sigmoid, :tanh or :linear' unless %i[sigmoid tanh linear].include?(output_activation)

        @input_size = input_size
        @hidden_size = hidden_size
        @output_size = output_size
        @output_activation = output_activation
        @seed = seed
        @rng = seed ? Random.new(seed) : Random.new
        init_weights
        @hidden_state = Array.new(hidden_size, 0.0)
      end

      # Current hidden state (a copy). Carries over between #eval calls
      # unless #reset_state is called or a new state is assigned.
      def hidden_state
        @hidden_state.dup
      end

      # Explicitly set the hidden state used by the next #eval call.
      def hidden_state=(state)
        raise ArgumentError, "expected #{@hidden_size} components" if state.length != @hidden_size

        @hidden_state = state.map(&:to_f)
      end

      # Reset the hidden state to zeros.
      def reset_state
        @hidden_state = Array.new(@hidden_size, 0.0)
      end

      # Run one sequence forward. +inputs+ is an array of input vectors.
      # Returns an array of output vectors, one per time step. The final
      # hidden state is kept in #hidden_state. Pass +initial_state+ to
      # override the starting state for this call.
      def eval(inputs, initial_state: nil)
        h = initial_state ? initial_state.map(&:to_f) : @hidden_state
        outputs = inputs.map do |x|
          check_vector(x, @input_size)
          h, cache = forward_step(x, h)
          cache[:y]
        end
        @hidden_state = h
        outputs
      end

      # Train with truncated backpropagation through time.
      #
      # +sequences+ and +targets+ are arrays of sequences (a batch). Each
      # sequence is an array of input (resp. target) vectors. Sequences in
      # the batch may have different lengths; +nil+ entries are treated as
      # padding: they neither update the hidden state nor contribute to
      # the loss.
      #
      # Every epoch, each sequence starts from a zero hidden state and is
      # processed in chunks of +truncation+ steps. The hidden state is
      # carried from chunk to chunk, but gradients never flow across a
      # chunk boundary. Returns the average loss per (non-padded) time
      # step for each epoch.
      def train(sequences, targets, epochs: 100, learning_rate: 0.1, truncation: 10)
        check_batch(sequences, targets)
        losses = []
        epochs.times do
          states = Array.new(sequences.size) { Array.new(@hidden_size, 0.0) }
          max_len = sequences.map(&:length).max
          total_loss = 0.0
          total_steps = 0
          (0...max_len).each_slice(truncation) do |range|
            grads = zero_gradients
            chunk_steps = 0
            sequences.each_with_index do |seq, s|
              inputs, tgts = chunk_inputs(seq, targets[s], range)
              next if inputs.empty?

              loss, seq_grads, final_state = loss_and_gradients(inputs, tgts, states[s])
              states[s] = final_state # carry state; gradient is detached here
              total_loss += loss * inputs.length
              total_steps += inputs.length
              chunk_steps += 1
              PARAMETER_KEYS.each { |k| add_gradients!(grads[k], seq_grads[k]) }
            end
            next if chunk_steps.zero?

            scale = learning_rate / sequences.size
            PARAMETER_KEYS.each { |k| apply_gradient!(@parameters[k], grads[k], scale) }
          end
          losses << (total_steps.zero? ? 0.0 : total_loss / total_steps)
        end
        losses
      end

      # Loss and gradients for a single contiguous chunk of one sequence.
      # Returns [loss, gradients, final_state]. +gradients+ has one entry
      # per parameter key plus :initial_state, the gradient with respect
      # to the chunk's initial hidden state (which #train deliberately
      # discards at truncation boundaries). Exposed mainly for testing
      # and custom training loops.
      def loss_and_gradients(inputs, targets, initial_state = nil)
        h = (initial_state || Array.new(@hidden_size, 0.0)).map(&:to_f)
        caches = []
        loss = 0.0
        inputs.each_with_index do |x, t|
          h, cache = forward_step(x, h)
          caches << cache
          loss += 0.5 * cache[:y].each_with_index.sum { |v, k| (v - targets[t][k])**2 }
        end
        steps = inputs.length
        loss /= steps

        grads = zero_gradients
        grads[:initial_state] = nil
        dh_next = Array.new(@hidden_size, 0.0)
        (steps - 1).downto(0) do |t|
          dh_next = backward_step(caches[t], targets[t], steps, dh_next, grads)
        end
        grads[:initial_state] = dh_next
        [loss, grads, h]
      end

      private

      def init_weights
        @parameters = {
          w_z: random_matrix(@input_size, @hidden_size),
          u_z: random_matrix(@hidden_size, @hidden_size),
          b_z: Array.new(@hidden_size, 0.0),
          w_r: random_matrix(@input_size, @hidden_size),
          u_r: random_matrix(@hidden_size, @hidden_size),
          b_r: Array.new(@hidden_size, 0.0),
          w_h: random_matrix(@input_size, @hidden_size),
          u_h: random_matrix(@hidden_size, @hidden_size),
          b_h: Array.new(@hidden_size, 0.0),
          w_y: random_matrix(@hidden_size, @output_size),
          b_y: Array.new(@output_size, 0.0)
        }
      end

      # Xavier-style uniform initialization, reproducible via +seed+.
      def random_matrix(rows, cols)
        limit = Math.sqrt(6.0 / (rows + cols))
        Array.new(rows) { Array.new(cols) { (@rng.rand * 2 * limit) - limit } }
      end

      def forward_step(x, h_prev)
        p = @parameters
        z = sigmoid_vec(add3(vec_mul_mat(x, p[:w_z]), vec_mul_mat(h_prev, p[:u_z]), p[:b_z]))
        r = sigmoid_vec(add3(vec_mul_mat(x, p[:w_r]), vec_mul_mat(h_prev, p[:u_r]), p[:b_r]))
        rh = hadamard(r, h_prev)
        hh = tanh_vec(add3(vec_mul_mat(x, p[:w_h]), vec_mul_mat(rh, p[:u_h]), p[:b_h]))
        h = Array.new(@hidden_size) { |i| ((1.0 - z[i]) * hh[i]) + (z[i] * h_prev[i]) }
        y = activate(add(vec_mul_mat(h, p[:w_y]), p[:b_y]))
        [h, { x: x, h_prev: h_prev, z: z, r: r, hh: hh, h: h, y: y }]
      end

      def backward_step(cache, target, steps, dh_next, grads)
        p = @parameters
        dy = cache[:y].each_with_index.map { |v, k| (v - target[k]) / steps }
        dy_pre = hadamard(dy, output_derivative(cache[:y]))

        accumulate_outer!(grads[:w_y], cache[:h], dy_pre)
        add_in_place!(grads[:b_y], dy_pre)
        dh = add(dh_next, mat_mul_vec(p[:w_y], dy_pre))

        dz = Array.new(@hidden_size) { |i| dh[i] * (cache[:h_prev][i] - cache[:hh][i]) }
        dhh = Array.new(@hidden_size) { |i| dh[i] * (1.0 - cache[:z][i]) }
        dh_prev = Array.new(@hidden_size) { |i| dh[i] * cache[:z][i] }

        dz_pre = Array.new(@hidden_size) { |i| dz[i] * cache[:z][i] * (1.0 - cache[:z][i]) }
        dhh_pre = Array.new(@hidden_size) { |i| dhh[i] * (1.0 - (cache[:hh][i]**2)) }

        drh = mat_mul_vec(p[:u_h], dhh_pre)
        dr = hadamard(drh, cache[:h_prev])
        add_in_place!(dh_prev, hadamard(drh, cache[:r]))
        dr_pre = Array.new(@hidden_size) { |i| dr[i] * cache[:r][i] * (1.0 - cache[:r][i]) }

        rh = hadamard(cache[:r], cache[:h_prev])
        accumulate_outer!(grads[:w_h], cache[:x], dhh_pre)
        accumulate_outer!(grads[:u_h], rh, dhh_pre)
        add_in_place!(grads[:b_h], dhh_pre)

        accumulate_outer!(grads[:w_r], cache[:x], dr_pre)
        accumulate_outer!(grads[:u_r], cache[:h_prev], dr_pre)
        add_in_place!(grads[:b_r], dr_pre)
        add_in_place!(dh_prev, mat_mul_vec(p[:u_r], dr_pre))

        accumulate_outer!(grads[:w_z], cache[:x], dz_pre)
        accumulate_outer!(grads[:u_z], cache[:h_prev], dz_pre)
        add_in_place!(grads[:b_z], dz_pre)
        add_in_place!(dh_prev, mat_mul_vec(p[:u_z], dz_pre))

        dh_prev
      end

      def activate(vec)
        case @output_activation
        when :sigmoid then sigmoid_vec(vec)
        when :tanh then tanh_vec(vec)
        else vec
        end
      end

      def output_derivative(y)
        case @output_activation
        when :sigmoid then y.map { |v| v * (1.0 - v) }
        when :tanh then y.map { |v| 1.0 - (v**2) }
        else Array.new(y.length, 1.0)
        end
      end

      def chunk_inputs(seq, tgts, range)
        inputs = []
        targets = []
        range.each do |t|
          x = seq[t]
          next if x.nil?

          inputs << x
          targets << tgts[t]
        end
        [inputs, targets]
      end

      def check_batch(sequences, targets)
        raise ArgumentError, 'batch must not be empty' if sequences.empty?
        raise ArgumentError, 'sequences and targets must have the same size' if sequences.size != targets.size

        sequences.each_with_index do |seq, s|
          raise ArgumentError, "sequence #{s} and its targets differ in length" if seq.length != targets[s].length

          seq.each { |x| check_vector(x, @input_size) if x }
          targets[s].each { |y| check_vector(y, @output_size) if y }
        end
      end

      def check_vector(vec, size)
        raise ArgumentError, "expected a vector of size #{size}" if vec.length != size
      end

      def zero_gradients
        grads = {}
        PARAMETER_KEYS.each do |k|
          grads[k] = @parameters[k].map { |row| row.is_a?(Array) ? Array.new(row.length, 0.0) : 0.0 }
        end
        grads
      end

      def add_gradients!(acc, grads)
        acc.each_index do |i|
          if acc[i].is_a?(Array)
            acc[i].each_index { |j| acc[i][j] += grads[i][j] }
          else
            acc[i] += grads[i]
          end
        end
      end

      def apply_gradient!(params, grads, scale)
        params.each_index do |i|
          if params[i].is_a?(Array)
            params[i].each_index { |j| params[i][j] -= scale * grads[i][j] }
          else
            params[i] -= scale * grads[i]
          end
        end
      end

      def accumulate_outer!(grad, a, b)
        grad.each_index do |i|
          ai = a[i]
          grad[i].each_index { |j| grad[i][j] += ai * b[j] }
        end
      end

      def vec_mul_mat(v, mat)
        cols = mat.first.length
        Array.new(cols) do |j|
          sum = 0.0
          v.each_index { |i| sum += v[i] * mat[i][j] }
          sum
        end
      end

      def mat_mul_vec(mat, v)
        mat.map do |row|
          sum = 0.0
          row.each_index { |j| sum += row[j] * v[j] }
          sum
        end
      end

      def add(a, b)
        a.each_index.map { |i| a[i] + b[i] }
      end

      def add3(a, b, c)
        a.each_index.map { |i| a[i] + b[i] + c[i] }
      end

      def add_in_place!(a, b)
        a.each_index { |i| a[i] += b[i] }
        a
      end

      def hadamard(a, b)
        a.each_index.map { |i| a[i] * b[i] }
      end

      def sigmoid_vec(vec)
        vec.map { |v| 1.0 / (1.0 + Math.exp(-v)) }
      end

      def tanh_vec(vec)
        vec.map { |v| Math.tanh(v) }
      end
    end
  end
end
