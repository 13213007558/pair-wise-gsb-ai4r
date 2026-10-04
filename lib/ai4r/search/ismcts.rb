# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
#
# Single-observer Information Set Monte Carlo Tree Search (SO-ISMCTS).

require_relative '../data/parameterizable'

module Ai4r
  module Search
    # Single-observer Information Set Monte Carlo Tree Search.
    #
    # Unlike plain MCTS, the search never holds a concrete opponent deal as
    # "the truth". Each iteration asks +sample_fn+ for a complete state that
    # is consistent with the observer's information, and tree statistics are
    # merged by information set identifier. Nodes therefore store only an
    # information set id and numeric per-action statistics; complete sampled
    # states are never retained, so hidden cards cannot leak through the tree
    # or through callback invocation order.
    #
    # An action may be legal in only some of the sampled states. Such actions
    # are tracked with two separate counters:
    #
    # - +availability+ counts how many sampled states allowed the action;
    # - +visits+ counts how often the action was actually selected.
    #
    # All payoffs returned by +payoff_fn+ must be expressed from the fixed
    # observer player's perspective (higher is better); no sign flipping based
    # on whose turn it is takes place.
    #
    # Callbacks (all receive working copies, never the caller's objects):
    #
    # - +sample_fn.call(root_state_copy, rng)+ returns a complete state that
    #   is consistent with the observation.
    # - +actions_fn.call(state)+ returns the legal actions; their order is
    #   irrelevant because the search sorts them deterministically.
    # - +transition_fn.call(state, action)+ returns the next state.
    # - +payoff_fn.call(state)+ returns the observer-perspective numeric
    #   payoff of a terminal state.
    # - +player_fn.call(state)+ returns whose turn it is.
    # - +info_set_fn.call(state)+ returns the information set identifier at an
    #   observer decision point.
    # - +terminal_fn.call(state)+ optionally detects terminal states; by
    #   default a state without legal actions is terminal.
    #
    # Example:
    #   searcher = Ai4r::Search::SingleObserverISMCTS.new(
    #     observer: :me,
    #     sample_fn: ->(observation, rng) { deal_consistently(observation, rng) },
    #     actions_fn: ->(state) { legal_actions(state) },
    #     transition_fn: ->(state, action) { apply(action, state) },
    #     payoff_fn: ->(state) { state.payoff_for(:me) },
    #     player_fn: ->(state) { state.turn },
    #     info_set_fn: ->(state) { state.information_set_id },
    #     seed: 1234
    #   )
    #   result = searcher.search(observed_state, 500, root_info_set: 'me:preflop')
    #   result.best_action
    class SingleObserverISMCTS
      include Ai4r::Data::Parameterizable

      # Per-action statistics attached to an information set node.
      ActionStats = Struct.new(:availability, :visits, :value_sum) do
        def initialize
          super(0, 0, 0.0)
        end

        def mean_value
          visits.zero? ? 0.0 : value_sum.to_f / visits
        end
      end

      # Tree node identified solely by an information set id. No complete
      # state is ever stored on a node.
      InfoSetNode = Struct.new(:id, :visits, :actions) do
        def initialize(id)
          super(id, 0, {})
        end
      end

      # Read-only view of the root decision. It exposes the chosen action and
      # the per-action counters but no sampled states or tree nodes.
      Result = Struct.new(:root_info_set, :root_visits, :best_action,
                          :action_visits, :action_availabilities,
                          :action_mean_values, keyword_init: true)

      parameters_info exploration: 'UCT exploration constant'

      # Create a new searcher.
      #
      # observer::       fixed player whose perspective is searched
      # sample_fn::       samples a complete state consistent with the view
      # actions_fn::      legal actions of a state
      # transition_fn::   next state after an action
      # payoff_fn::       observer-perspective terminal payoff
      # player_fn::       player to move in a state
      # info_set_fn::     information set id at observer decision states
      # terminal_fn::     optional terminal predicate
      # exploration::    UCT exploration constant
      # seed::            optional fixed seed for deterministic searches
      # rng::             optional random source (overrides +seed+)
      # copy_fn::         optional deep-copy strategy for mutable states
      # action_key_fn::   optional sort key used to order actions safely
      def initialize(observer:, sample_fn:, actions_fn:, transition_fn:,
                     payoff_fn:, player_fn:, info_set_fn:, terminal_fn: nil,
                     exploration: Math.sqrt(2), seed: nil, rng: nil,
                     copy_fn: nil, action_key_fn: nil)
        @observer = observer
        @sample_fn = sample_fn
        @actions_fn = actions_fn
        @transition_fn = transition_fn
        @payoff_fn = payoff_fn
        @player_fn = player_fn
        @info_set_fn = info_set_fn
        @terminal_fn = terminal_fn
        @exploration = exploration
        @rng = rng || (seed ? Random.new(seed) : Random.new)
        @copy_fn = copy_fn || method(:default_deep_copy)
        @action_key_fn = action_key_fn || ->(action) { action }
      end

      # Run the search for +iterations+ simulations starting from
      # +root_state+. The caller supplies the +root_info_set+ identifier that
      # describes what the observer currently knows; it is the same value on
      # every call made from an equivalent observation.
      #
      # Neither +root_state+ nor any mutable object returned by +sample_fn+ is
      # modified: both are deep copied before the search touches them.
      def search(root_state, iterations, root_info_set:)
        @nodes = {}
        root_node = node(root_info_set)
        iterations.times do
          sampled = @sample_fn.call(deep_copy(root_state), @rng)
          state = deep_copy(sampled)
          path = []
          starting_node = observer_turn?(state) ? root_node : nil
          payoff = select_and_expand(state, starting_node, path)
          backup(path, payoff)
        end
        build_result(root_info_set)
      end

      private

      def select_and_expand(state, node, path)
        loop do
          return @payoff_fn.call(state) if terminal?(state)

          actions = legal_actions(state)
          return @payoff_fn.call(state) if actions.empty?

          if @player_fn.call(state) == @observer
            node ||= node(@info_set_fn.call(state))
            mark_availability(node, actions)

            untried = actions.reject do |action|
              (node.actions[action]&.visits || 0).positive?
            end
            if untried.any?
              action = random_action(untried)
              path << [node, action]
              next_state = @transition_fn.call(state, action)
              return rollout(next_state)
            end

            action = best_ucb_action(node, actions)
            path << [node, action]
            state = @transition_fn.call(state, action)
            node = nil
          else
            state = @transition_fn.call(state, random_action(actions))
          end
        end
      end

      def rollout(state)
        loop do
          return @payoff_fn.call(state) if terminal?(state)

          actions = legal_actions(state)
          return @payoff_fn.call(state) if actions.empty?

          state = @transition_fn.call(state, random_action(actions))
        end
      end

      def mark_availability(node, actions)
        actions.each do |action|
          node.actions[action] ||= ActionStats.new
          node.actions[action].availability += 1
        end
      end

      def best_ucb_action(node, actions)
        log_visits = Math.log(node.visits)
        actions.max_by do |action|
          stats = node.actions.fetch(action)
          exploitation = stats.mean_value
          exploration = @exploration *
                         Math.sqrt(log_visits / stats.visits)
          exploitation + exploration
        end
      end

      def backup(path, payoff)
        path.each do |node, action|
          stats = node.actions.fetch(action)
          stats.visits += 1
          stats.value_sum += payoff
          node.visits += 1
        end
      end

      def build_result(root_info_set)
        root_node = @nodes.fetch(root_info_set)
        ordered = root_node.actions.sort_by { |action, _| @action_key_fn.call(action) }
        visits = {}
        availabilities = {}
        mean_values = {}
        ordered.each do |action, stats|
          visits[action] = stats.visits
          availabilities[action] = stats.availability
          mean_values[action] = stats.mean_value
        end
        best_action = ordered.max_by { |_, stats| stats.visits }&.first
        Result.new(
          root_info_set: root_info_set,
          root_visits: root_node.visits,
          best_action: best_action,
          action_visits: visits,
          action_availabilities: availabilities,
          action_mean_values: mean_values
        )
      end

      def legal_actions(state)
        @actions_fn.call(state).sort_by { |action| @action_key_fn.call(action) }
      end

      def observer_turn?(state)
        !terminal?(state) && @player_fn.call(state) == @observer
      end

      def terminal?(state)
        @terminal_fn ? @terminal_fn.call(state) : false
      end

      def random_action(actions)
        actions[@rng.rand(actions.length)]
      end

      def node(info_set_id)
        @nodes[info_set_id] ||= InfoSetNode.new(info_set_id)
      end

      def deep_copy(state)
        @copy_fn.call(state)
      end

      def default_deep_copy(state)
        Marshal.load(Marshal.dump(state))
      end
    end
  end
end
