# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/search/ismcts'

class ISMCTSTest < Minitest::Test
  include Ai4r::Search

  # Hidden-card mini game with only 8 complete states:
  #   2 roots  (opponent holds :low or :high, hidden from the observer)
  #   6 terminals (one per opponent card x :check/:bet/:bluff)
  #
  # The observer always holds :mid. Public information is just "no action
  # taken yet", so both roots share the information set :root.
  #
  # Actions:
  #   :check -> 0.0
  #   :bet   -> +2.0 vs :low, -1.0 vs :high
  #   :bluff -> +3.0, but only legal when the opponent holds :high
  def build_game
    {
      info_set_fn: ->(s) { s[:action] ? [:done, s[:action]] : :root },
      actions_fn: ->(s) { s[:opp] == :high ? %i[check bet bluff] : %i[check bet] },
      transition_fn: ->(s, a) { { opp: s[:opp], action: a } },
      terminal_fn: ->(s) { !s[:action].nil? },
      reward_fn: lambda { |s|
        case s[:action]
        when :check then 0.0
        when :bet then s[:opp] == :low ? 2.0 : -1.0
        when :bluff then 3.0
        end
      }
    }
  end

  # Deterministic sampler: always replays the same sequence of complete
  # states, driven only by the (public) information set identifier.
  def sequence_sampler(states)
    index = 0
    lambda do |_info_set|
      state = states[index % states.length]
      index += 1
      state
    end
  end

  def test_same_public_info_yields_same_root_decision
    sequence = [{ opp: :low, action: nil },
                { opp: :low, action: nil },
                { opp: :high, action: nil }]
    results = %i[low high].map do |true_opp|
      ismcts = ISMCTS.new(**build_game,
                          sample_fn: sequence_sampler(sequence),
                          random: Random.new(7))
      decision = ismcts.search({ opp: true_opp, action: nil }, 200)
      [decision, ismcts.statistics]
    end
    assert_equal results.first, results.last
    assert_equal :bet, results.first.first
  end

  def test_availability_and_visits_are_tracked_separately
    sequence = [{ opp: :low, action: nil }, { opp: :high, action: nil }]
    ismcts = ISMCTS.new(**build_game,
                        sample_fn: sequence_sampler(sequence),
                        random: Random.new(3))
    ismcts.search({ opp: :high, action: nil }, 40)
    stats = ismcts.statistics

    assert_equal 40, ismcts.root.visits
    assert_equal 40, stats[:check][:availability]
    assert_equal 40, stats[:bet][:availability]
    assert_equal 20, stats[:bluff][:availability]
    total_visits = stats.values.sum { |s| s[:visits] }
    assert_equal 40, total_visits
    assert_in_delta 3.0, stats[:bluff][:mean]
    assert_in_delta 0.0, stats[:check][:mean]
  end

  def test_search_does_not_mutate_input_or_sampled_states
    root_state = { opp: :high, action: nil, log: [] }
    samples = [{ opp: :low, action: nil, log: [] },
               { opp: :high, action: nil, log: [] }]
    root_before = Marshal.load(Marshal.dump(root_state))
    samples_before = Marshal.load(Marshal.dump(samples))

    ismcts = ISMCTS.new(**build_game,
                        sample_fn: sequence_sampler(samples),
                        random: Random.new(5))
    ismcts.search(root_state, 60)

    assert_equal root_before, root_state
    assert_equal samples_before, samples
  end

  def test_nodes_store_no_full_states
    sequence = [{ opp: :low, action: nil }, { opp: :high, action: nil }]
    ismcts = ISMCTS.new(**build_game,
                        sample_fn: sequence_sampler(sequence),
                        random: Random.new(11))
    ismcts.search({ opp: :low, action: nil }, 50)

    assert_equal :root, ismcts.root.info_set
    refute ismcts.root.instance_variables.include?(:@state)
    ismcts.root.children.each_value do |child|
      refute child.instance_variables.include?(:@state)
    end
  end
end
