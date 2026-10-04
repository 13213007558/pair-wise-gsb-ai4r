# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
#
# Q-learning with a learned MDP model and prioritized sweeping planning.

require 'set'
require_relative '../data/parameterizable'

module Ai4r
  module Reinforcement
    # Tabular Dyna-style agent that stores every observed transition in a
    # model and runs prioritized sweeping backups. A newly discovered reward
    # therefore propagates backwards through previously observed predecessors
    # instead of waiting for the agent to physically revisit those states.
    #
    # Interaction updates (one Q-learning backup per real observation) and
    # model-based planning updates are counted separately, and the caller
    # controls how many planning updates each observation may trigger.
    class PrioritizedSweepingAgent
      include Ai4r::Data::Parameterizable

      parameters_info learning_rate: 'Update step size',
                      discount: 'Discount factor',
                      exploration: 'Exploration rate',
                      threshold: 'Minimum priority worth sweeping'

      # @param rng [Random] source of randomness for exploration and model sampling
      def initialize(rng: Random.new)
        @learning_rate = 0.1
        @discount = 0.9
        @exploration = 0.1
        @threshold = 0.0
        @rng = rng

        @q = Hash.new { |h, k| h[k] = Hash.new(0.0) }
        # model[state][action] => { [next_state, terminal] => { count:, rewards: [] } }
        @model = {}
        @predecessors = Hash.new { |h, k| h[k] = Set.new }
        @tried_actions = Hash.new { |h, k| h[k] = Set.new }
        @legal_actions = {}
        @priority = {}
        @heap = []
        @insertion = 0
        @interaction_updates = 0
        @planning_updates = 0
      end

      # Record one observed transition, perform the single Q-learning backup,
      # enqueue the state-action by its temporal-difference error, and
      # optionally run prioritized sweeping.
      #
      # @param state [Object] state the action was taken in
      # @param action [Object] action taken
      # @param reward [Numeric] reward received
      # @param next_state [Object] resulting state
      # @param terminal [Boolean] true when next_state is terminal
      # @param available_actions [Array, nil] actions available in next_state
      # @param planning_budget [Integer, nil] maximum model-based updates
      # @return [Array] state-actions backed up during planning
      def update(state, action, reward, next_state, terminal: false,
                 available_actions: nil, planning_budget: nil)
        register_actions(next_state, available_actions)
        remember(state, action, reward, next_state, terminal)

        target = terminal ? reward.to_f : reward.to_f + @discount * value(next_state)
        previous_q = @q[state][action]
        q_learn(state, action, target)
        @interaction_updates += 1

        prioritize(state, action, (target - previous_q).abs)
        plan(planning_budget) if planning_budget
      end

      # Run up to +budget+ prioritized sweeping backups. Each backup pops the
      # currently highest-priority state-action, samples one stored outcome,
      # and refreshes the priorities of its known predecessors.
      #
      # Stale heap entries (superseded priorities or state-actions no longer
      # worth sweeping) are discarded without consuming the caller's budget.
      #
      # @return [Array] state-actions actually backed up
      def plan(budget = 1)
        swept = []
        budget.to_i.times do
          state, action = pop_ready_entry
          break unless state

          reward, next_state, terminal = sample_outcome(state, action)
          target = terminal ? reward : reward + @discount * value(next_state)
          q_learn(state, action, target)
          @planning_updates += 1
          swept << [state, action]

          @predecessors[state].each do |predecessor, previous_action|
            predecessor_reward, predecessor_next, predecessor_terminal =
              sample_outcome(predecessor, previous_action)
            predecessor_target =
              if predecessor_terminal
                predecessor_reward
              else
                predecessor_reward + @discount * value(predecessor_next)
              end
            residual = (predecessor_target - @q[predecessor][previous_action]).abs
            prioritize(predecessor, previous_action, residual)
          end
        end
        swept
      end

      # Choose an action using an epsilon-greedy strategy over the actions
      # currently known to be available in +state+.
      def choose_action(state)
        actions = known_actions(state)
        return nil if actions.empty?

        if @rng.rand < @exploration
          actions[@rng.rand(actions.length)]
        else
          best = actions.map { |action| @q[state][action] }.max
          best_actions = actions.select { |action| @q[state][action] == best }
          best_actions[@rng.rand(best_actions.length)]
        end
      end

      # State value: max Q over actions available in +state+. A state with no
      # available actions (terminal or never observed) is worth zero, so
      # unavailable or untried actions can never enter the maximum.
      def value(state)
        actions = known_actions(state)
        return 0.0 if actions.empty?

        actions.map { |action| @q[state][action] }.max.to_f
      end

      def queue_size
        @heap.length
      end

      # Number of state-actions currently waiting to be swept (stale heap
      # copies are excluded).
      def pending_queue_size
        @priority.length
      end

      # Declare which actions are legal in +state+. Only declared actions
      # participate in the max used for bootstrapping and action selection.
      # Pass an empty array for terminal states.
      def declare_available_actions(state, actions)
        register_actions(state, actions)
      end

      def queued_priority(state, action)
        token = @priority[[state, action]]
        token && token[:priority]
      end

      def observed_outcomes(state, action)
        return [] unless @model[state] && @model[state][action]

        @model[state][action].map do |(next_state, terminal), data|
          { next_state: next_state, terminal: terminal,
            count: data[:count], rewards: data[:rewards].dup }
        end
      end

      attr_reader :q, :rng, :interaction_updates, :planning_updates

      private

      def register_actions(state, actions)
        return if state.nil? || actions.nil?

        @legal_actions[state] = actions.to_a.uniq
      end

      def known_actions(state)
        if @legal_actions.key?(state)
          @legal_actions[state]
        else
          @tried_actions[state].to_a
        end
      end

      def remember(state, action, reward, next_state, terminal)
        @tried_actions[state] << action
        @predecessors[next_state] << [state, action] unless next_state.nil?

        @model[state] ||= {}
        outcomes = @model[state][action] ||= Hash.new do |hash, key|
          hash[key] = { count: 0, rewards: [] }
        end
        outcome = outcomes[[next_state, terminal]]
        outcome[:count] += 1
        outcome[:rewards] << reward.to_f
      end

      def q_learn(state, action, target)
        @q[state][action] += @learning_rate * (target - @q[state][action])
      end

      def prioritize(state, action, residual)
        return if residual <= @threshold

        key = [state, action]
        pending = @priority[key]
        # A queued copy only needs replacing for a larger residual; the old
        # heap token then becomes stale and is discarded on pop. Once popped
        # (no pending token), any super-threshold residual starts a new wave.
        return if pending && pending[:priority] >= residual

        @insertion += 1
        token = { key: key, priority: residual, insertion: @insertion }
        @priority[key] = token
        @heap << token
      end

      def pop_ready_entry
        while (entry = pop_heap)
          key = entry[:key]
          next unless @priority[key].equal?(entry)

          @priority.delete(key)
          return key
        end
        nil
      end

      def pop_heap
        return nil if @heap.empty?

        last = @heap.pop
        return last if @heap.empty?

        root = @heap[0]
        @heap[0] = last
        sift_down(0)
        root
      end

      def sift_down(index)
        loop do
          left = index * 2 + 1
          break if left >= @heap.length

          right = left + 1
          child =
            if right < @heap.length && higher_priority?(@heap[right], @heap[left])
              right
            else
              left
            end
          break unless higher_priority?(@heap[child], @heap[index])

          @heap[index], @heap[child] = @heap[child], @heap[index]
          index = child
        end
      end

      # Larger residual first; ties keep the earlier insertion (FIFO), which
      # follows the discovery order of the reward backwards along a corridor.
      def higher_priority?(first, second)
        first[:priority] > second[:priority] ||
          (first[:priority] == second[:priority] &&
           first[:insertion] < second[:insertion])
      end

      def sample_outcome(state, action)
        outcomes = @model.dig(state, action)
        return [0.0, nil, true] unless outcomes

        total = outcomes.sum { |_, data| data[:count] }
        draw = @rng.rand(total)
        outcomes.each do |(next_state, terminal), data|
          draw -= data[:count]
          if draw.negative?
            return [data[:rewards][@rng.rand(data[:rewards].length)],
                    next_state, terminal]
          end
        end
      end
    end
  end
end
