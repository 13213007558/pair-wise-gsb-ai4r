# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
#
# Single-observer Information Set Monte Carlo Tree Search (ISMCTS).

require_relative '../data/parameterizable'

module Ai4r
  module Search
    # Single-observer ISMCTS for games with hidden information.
    #
    # Unlike plain MCTS, the search tree is built over *information sets*
    # (what the observing player can see) instead of full game states, so
    # the opponent's hidden cards are never used as a basis for decisions.
    #
    # The caller provides six callbacks:
    # - +info_set_fn.call(state)+ returns a public identifier for the
    #   information set of +state+ (everything the observer may know, and
    #   nothing more). Statistics of all sampled states mapping to the same
    #   identifier are merged into a single node.
    # - +sample_fn.call(info_set)+ returns a full state consistent with the
    #   observer's information (a determinization). It is the only source of
    #   hidden information and is driven solely by the public identifier.
    # - +actions_fn.call(state)+ lists the legal actions for a sampled state.
    #   An action may be legal in only some sampled states; the search tracks
    #   an availability count per action and only selects among actions that
    #   are legal in the current sample.
    # - +transition_fn.call(state, action)+ computes the next state.
    # - +terminal_fn.call(state)+ detects terminal states.
    # - +reward_fn.call(state)+ returns the payoff of a terminal state from
    #   the designated (observing) player's perspective. The search never
    #   negates or reinterprets it.
    #
    # Privacy and integrity guarantees:
    # - Tree nodes store only the public information-set identifier, the
    #   incoming action and aggregate statistics. Full sampled states exist
    #   only as local variables during a single iteration, so hidden
    #   information cannot leak through stored node state.
    # - The search never mutates the root state, sampled states, or any
    #   object returned by the callbacks; callbacks are expected to be pure.
    #
    # Example:
    #   ismcts = Ai4r::Search::ISMCTS.new(
    #     info_set_fn: ->(s) { s[:public] },
    #     sample_fn: ->(info) { deal_consistent_with(info) },
    #     actions_fn: ->(s) { legal_actions(s) },
    #     transition_fn: ->(s, a) { apply(s, a) },
    #     terminal_fn: ->(s) { s[:over] },
    #     reward_fn: ->(s) { s[:my_score] }
    #   )
    #   best = ismcts.search(current_state, 100)
    #   ismcts.statistics # => { action => {visits:, value:, mean:, availability:} }
    class ISMCTS
      include Ai4r::Data::Parameterizable

      # A tree node: one information set reached by a public action history.
      # Deliberately holds no full game state.
      class Node
        attr_reader :info_set, :action, :parent, :children, :availability
        attr_accessor :visits, :value

        def initialize(info_set, parent = nil, action = nil)
          @info_set = info_set
          @parent = parent
          @action = action
          @children = {}
          @visits = 0
          @value = 0.0
          @availability = Hash.new(0)
        end
      end

      parameters_info exploration: 'UCT exploration constant'

      # Create a new single-observer ISMCTS search object.
      #
      # info_set_fn::   public information-set identifier for a state
      # sample_fn::     samples a full state consistent with an information set
      # actions_fn::    legal actions for a sampled state
      # transition_fn:: next state given a state and an action
      # terminal_fn::   predicate to detect terminal states
      # reward_fn::     payoff of a terminal state, from the designated
      #                 (observing) player's perspective
      # exploration::   UCT exploration constant
      # random::        random number generator used for tie-breaking and
      #                 rollout policy (inject a seeded generator for
      #                 reproducible searches)
      def initialize(info_set_fn:, sample_fn:, actions_fn:, transition_fn:,
                     terminal_fn:, reward_fn:,
                     exploration: Math.sqrt(2), random: Random.new)
        @info_set_fn = info_set_fn
        @sample_fn = sample_fn
        @actions_fn = actions_fn
        @transition_fn = transition_fn
        @terminal_fn = terminal_fn
        @reward_fn = reward_fn
        @exploration = exploration
        @random = random
        @root = nil
      end

      # The root information-set node of the last search (nil before the
      # first search). Exposes aggregated visits and availability counts.
      attr_reader :root

      # Run ISMCTS from the information set of +root_state+ for a number of
      # +iterations+. Returns the most visited root action.
      #
      # +root_state+ is only used to compute the public information-set
      # identifier; its hidden components are never inspected and the object
      # itself is never modified.
      def search(root_state, iterations)
        @root = Node.new(@info_set_fn.call(root_state))
        iterations.times do
          sampled = @sample_fn.call(@root.info_set)
          node, leaf_state = tree_policy(@root, sampled)
          reward = default_policy(leaf_state)
          backup(node, reward)
        end
        best_root_action
      end

      # Statistics of the root information set from the last search, keyed
      # by action: visits and value of the child node plus the number of
      # iterations in which the action was legal in the sampled state.
      def statistics
        return {} unless @root

        @root.children.each_with_object({}) do |(action, child), stats|
          stats[action] = {
            visits: child.visits,
            value: child.value,
            mean: child.visits.zero? ? 0.0 : child.value / child.visits,
            availability: @root.availability[action]
          }
        end
      end

      private

      def tree_policy(node, state)
        until @terminal_fn.call(state)
          legal = @actions_fn.call(state)
          legal.each { |action| node.availability[action] += 1 }
          untried = legal - node.children.keys
          return expand(node, state, untried) unless untried.empty?

          child = best_child(node, legal)
          state = @transition_fn.call(state, child.action)
          node = child
        end
        [node, state]
      end

      def expand(node, state, untried)
        action = untried.sample(random: @random)
        child_state = @transition_fn.call(state, action)
        child = Node.new(@info_set_fn.call(child_state), node, action)
        node.children[action] = child
        [child, child_state]
      end

      # UCT selection restricted to actions legal in the current sample, so
      # actions that are only sometimes legal never get forced.
      def best_child(node, legal)
        candidates = node.children.values.select { |c| legal.include?(c.action) }
        candidates.max_by do |child|
          exploitation = child.value / child.visits
          exploration = @exploration * Math.sqrt(Math.log(node.visits) / child.visits)
          exploitation + exploration
        end
      end

      def default_policy(state)
        current = state
        until @terminal_fn.call(current)
          action = @actions_fn.call(current).sample(random: @random)
          current = @transition_fn.call(current, action)
        end
        @reward_fn.call(current)
      end

      def backup(node, reward)
        while node
          node.visits += 1
          node.value += reward
          node = node.parent
        end
      end

      def best_root_action
        return nil if @root.children.empty?

        @root.children.values.max_by(&:visits).action
      end
    end
  end
end
