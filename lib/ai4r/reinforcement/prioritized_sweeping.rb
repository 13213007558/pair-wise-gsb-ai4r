# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
#
# Q-learning with a learned sample model and prioritized sweeping.

require_relative '../data/parameterizable'

module Ai4r
  module Reinforcement
    # Planning agent combining Q-learning with prioritized sweeping.
    #
    # Each observation performs a direct TD update (an *interaction
    # update*) and is stored in a sample model that allows the same
    # state-action pair to produce different successors and rewards.
    # A priority queue of state-action pairs is then processed inside a
    # caller-provided planning budget, so newly discovered rewards
    # propagate backwards along already observed predecessors
    # (*planning updates*). Interaction and planning updates are
    # counted separately.
    #
    #   agent = Ai4r::Reinforcement::PrioritizedSweeping.new
    #   agent.update(:s1, :go, 1, :goal, done: true, planning_budget: 10)
    #   agent.plan(5) # keep planning later
    class PrioritizedSweeping
      include Ai4r::Data::Parameterizable

      parameters_info learning_rate: 'Update step size',
                      discount: 'Discount factor',
                      exploration: 'Exploration rate',
                      threshold: 'Minimum priority to keep a pair queued'

      attr_reader :interaction_updates, :planning_updates, :q, :model

      def initialize
        @learning_rate = 0.1
        @discount = 0.9
        @exploration = 0.1
        @threshold = 1e-6
        @q = Hash.new { |hash, key| hash[key] = Hash.new(0.0) }
        @model = Hash.new { |hash, key| hash[key] = Hash.new { |h, k| h[k] = Hash.new(0) } }
        @predecessors = Hash.new { |hash, key| hash[key] = {} }
        @terminal = {}
        @queue = []
        @sequence = 0
        @interaction_updates = 0
        @planning_updates = 0
      end

      # Observe a transition and spend up to +planning_budget+ model-based
      # updates. Returns the list of [state, action] pairs updated during
      # planning, in the order they were processed.
      def update(state, action, reward, next_state, done: false, planning_budget: 0)
        record_transition(state, action, reward, next_state, done)
        @q[state][action] += @learning_rate *
                             (reward + @discount * state_value(next_state) - @q[state][action])
        @interaction_updates += 1
        enqueue(state, action)
        each_predecessor(state) { |pred_state, pred_action| enqueue(pred_state, pred_action) }
        plan(planning_budget)
      end

      # Spend up to +budget+ planning updates from the priority queue.
      # Stale entries whose recomputed priority no longer exceeds the
      # threshold are discarded without consuming budget. Returns the
      # list of [state, action] pairs updated, in processing order.
      def plan(budget)
        updated = []
        while updated.length < budget
          entry = pop_relevant
          break unless entry

          state, action = entry
          @q[state][action] += @learning_rate *
                               (expected_target(state, action) - @q[state][action])
          @planning_updates += 1
          updated << [state, action]
          each_predecessor(state) { |pred_state, pred_action| enqueue(pred_state, pred_action) }
        end
        updated
      end

      # Choose an action among the actions observed from +state+ using an
      # ε-greedy strategy. Pass a Random instance for reproducible draws.
      def choose_action(state, random: Random)
        actions = @model.fetch(state, nil)
        return nil if actions.nil? || actions.empty?

        keys = actions.keys
        if random.rand < @exploration
          keys.sample(random: random)
        else
          keys.max_by { |action| q_value(state, action) }
        end
      end

      # Number of entries currently sitting in the priority queue,
      # including not-yet-discarded stale ones.
      def queue_size
        @queue.size
      end

      private

      def record_transition(state, action, reward, next_state, done)
        @model[state][action][[next_state, reward]] += 1
        @predecessors[next_state][[state, action]] = true
        @terminal[next_state] = true if done
      end

      def q_value(state, action)
        actions = @q.fetch(state, nil)
        actions ? actions[action] : 0.0
      end

      # Value of a state: zero for terminal states, otherwise the maximum
      # Q-value over actions actually observed from that state.
      def state_value(state)
        return 0.0 if @terminal[state]

        actions = @model.fetch(state, nil)
        return 0.0 if actions.nil? || actions.empty?

        actions.keys.map { |action| q_value(state, action) }.max
      end

      def outcomes_for(state, action)
        actions = @model.fetch(state, nil)
        actions ? actions.fetch(action, {}) : {}
      end

      # Model-based expectation of reward + discount * value(successor).
      def expected_target(state, action)
        outcomes = outcomes_for(state, action)
        total = outcomes.values.sum
        return q_value(state, action) if total.zero?

        outcomes.sum do |(next_state, reward), count|
          count * (reward + @discount * state_value(next_state))
        end / total.to_f
      end

      def priority_of(state, action)
        (expected_target(state, action) - q_value(state, action)).abs
      end

      def each_predecessor(state)
        predecessors = @predecessors.fetch(state, nil)
        return unless predecessors

        predecessors.each_key { |pred_state, pred_action| yield pred_state, pred_action }
      end

      def enqueue(state, action)
        priority = priority_of(state, action)
        return if priority <= @threshold

        @sequence += 1
        push([priority, -@sequence, state, action])
      end

      def pop_relevant
        until @queue.empty?
          _priority, _neg_sequence, state, action = pop
          return [state, action] if priority_of(state, action) > @threshold
        end
        nil
      end

      def push(entry)
        @queue << entry
        index = @queue.size - 1
        while index.positive?
          parent = (index - 1) / 2
          break if (@queue[parent] <=> @queue[index]) >= 0

          @queue[parent], @queue[index] = @queue[index], @queue[parent]
          index = parent
        end
      end

      def pop
        top = @queue[0]
        last = @queue.pop
        unless @queue.empty?
          @queue[0] = last
          index = 0
          loop do
            left = 2 * index + 1
            right = left + 1
            largest = index
            largest = left if left < @queue.size && (@queue[left] <=> @queue[largest]).positive?
            largest = right if right < @queue.size && (@queue[right] <=> @queue[largest]).positive?
            break if largest == index

            @queue[index], @queue[largest] = @queue[largest], @queue[index]
            index = largest
          end
        end
        top
      end
    end
  end
end
