# frozen_string_literal: true

require_relative '../test_helper'
require 'ai4r/genetic_algorithm/genetic_algorithm'
require_relative '../helpers/genetic/bit_chromosome'

module Ai4r
  module GeneticAlgorithm
    class StaticRouteChromosome < ChromosomeBase
      attr_reader :label

      def initialize(label, objectives)
        @label = label
        super(objectives)
      end

      def fitness
        data
      end
    end

    class EnumerableRouteChromosome < ChromosomeBase
      VALUES = (0..4).to_a.freeze

      attr_accessor :value

      def initialize(value = rand(VALUES.length))
        self.value = value
        super([value, VALUES.length - 1 - value])
      end

      def fitness
        data
      end

      def self.seed
        new
      end

      def self.reproduce(first, second, crossover_rate)
        if rand < crossover_rate
          new((first.value + second.value) / 2)
        else
          new(first.value)
        end
      end

      def self.mutate(chromosome, mutation_rate)
        return unless rand < mutation_rate

        shift = rand < 0.5 ? -1 : 1
        chromosome.value = [[chromosome.value + shift, 0].max, VALUES.length - 1].min
        chromosome.data = [chromosome.value, VALUES.length - 1 - chromosome.value]
      end
    end

    class GuardedRouteChromosome < ChromosomeBase
      OBJECTIVES = [[0, 3], [1, 2], [2, 1]].freeze

      def initialize(objectives = nil)
        super(objectives || OBJECTIVES.sample.dup)
      end

      def fitness
        data
      end

      def self.seed
        new
      end

      def self.reproduce(first, _second, _crossover_rate)
        new(first.data.dup)
      end

      def self.mutate(chromosome, _mutation_rate)
        chromosome.data.map! { |value| value + 100 }
      end
    end

    class Nsga2GeneticSearchTest < Minitest::Test
      def test_non_dominated_sort_uses_dominance_and_keeps_front_order
        individuals = [
          StaticRouteChromosome.new(:a, [0, 2]),
          StaticRouteChromosome.new(:b, [1, 1]),
          StaticRouteChromosome.new(:c, [2, 0]),
          StaticRouteChromosome.new(:d, [1, 2]),
          StaticRouteChromosome.new(:e, [2, 2])
        ]
        search = build_search(individuals.length, objectives: %i[minimize minimize])

        fronts = search.non_dominated_sort(individuals)

        assert_equal([%i[a b c], %i[d], %i[e]], fronts.map { |front| front.map(&:label) })
      end

      def test_objective_directions_can_mix_minimization_and_maximization
        individuals = [
          StaticRouteChromosome.new(:a, [0, 0]),
          StaticRouteChromosome.new(:b, [1, 1]),
          StaticRouteChromosome.new(:c, [2, 2]),
          StaticRouteChromosome.new(:dominated, [1, 0])
        ]
        search = build_search(individuals.length, objectives: %i[minimize maximize])

        front = search.pareto_front(individuals)

        assert_equal %i[a b c], front.map(&:label)
      end

      def test_environmental_selection_uses_fronts_then_crowding_distance
        individuals = [
          StaticRouteChromosome.new(:a, [0, 4]),
          StaticRouteChromosome.new(:b, [1, 3]),
          StaticRouteChromosome.new(:c, [2, 2]),
          StaticRouteChromosome.new(:d, [3, 1]),
          StaticRouteChromosome.new(:e, [4, 0]),
          StaticRouteChromosome.new(:dominated, [4, 4])
        ]
        search = build_search(4, objectives: %i[minimize minimize])
        ranked = search.rank_population(individuals)

        selected = search.environmental_selection(ranked).map { |entry| entry.individual.label }

        assert_equal %i[a e c b], selected
      end

      def test_duplicate_boundary_and_constant_objective_ties_are_stable
        duplicated_boundaries = [
          StaticRouteChromosome.new(:a, [0, 3]),
          StaticRouteChromosome.new(:b, [0, 3]),
          StaticRouteChromosome.new(:c, [1, 2]),
          StaticRouteChromosome.new(:d, [2, 1]),
          StaticRouteChromosome.new(:e, [3, 0])
        ]
        search = build_search(3, objectives: %i[minimize minimize])

        selected = search.environmental_selection(search.rank_population(duplicated_boundaries))

        assert_equal(%i[a b e], selected.map { |entry| entry.individual.label })

        constant_objective = [
          StaticRouteChromosome.new(:f, [0, 1]),
          StaticRouteChromosome.new(:g, [0, 1]),
          StaticRouteChromosome.new(:h, [1, 1]),
          StaticRouteChromosome.new(:i, [2, 1])
        ]
        search = build_search(3, objectives: %i[minimize minimize])

        selected = search.environmental_selection(search.rank_population(constant_objective))

        assert_equal(%i[f g h], selected.map { |entry| entry.individual.label })
      end

      def test_identical_objective_vectors_keep_population_order
        individuals = [
          StaticRouteChromosome.new(:a, [1, 1]),
          StaticRouteChromosome.new(:b, [1, 1]),
          StaticRouteChromosome.new(:c, [1, 1]),
          StaticRouteChromosome.new(:d, [1, 1])
        ]
        search = build_search(3, objectives: %i[minimize minimize])

        selected = search.environmental_selection(search.rank_population(individuals))

        assert_equal(%i[a b c], selected.map { |entry| entry.individual.label })
      end

      def test_reproduction_does_not_mutate_selected_parent_individuals
        originals = GuardedRouteChromosome::OBJECTIVES.map do |objectives|
          GuardedRouteChromosome.new(objectives.dup.freeze)
        end
        search = build_search(originals.length, 1, GuardedRouteChromosome,
                              objectives: %i[minimize minimize],
                              mutation_rate: 1.0, crossover_rate: 0.0)
        ranked = search.rank_population(originals)

        offspring = search.send(:reproduce_multi_objective, ranked)

        assert_equal [[0, 3], [1, 2], [2, 1]], originals.map(&:fitness)
        assert(offspring.none? { |child| originals.include?(child) })
      end

      def test_fixed_random_run_returns_an_undominated_pareto_front
        srand(240_210)
        search = build_search(8, 4, EnumerableRouteChromosome, objectives: %i[minimize minimize],
                                                               mutation_rate: 0.2, crossover_rate: 0.7)

        front = search.run
        final_candidates = search.population

        assert_equal 8, final_candidates.length
        assert(front.length.positive?)
        assert(front.all? { |individual| nondominated?(individual, final_candidates) })
      end

      def test_scalar_run_remains_the_legacy_interface
        srand(7)
        search = GeneticSearch.new(4, 1, BitChromosome)

        result = search.run

        assert_kind_of ChromosomeBase, result
      end

      private

      def build_search(population_size, generations = 1, chromosome_class = StaticRouteChromosome, **options)
        GeneticSearch.new(population_size, generations, chromosome_class,
                          options.fetch(:mutation_rate, 0.2),
                          options.fetch(:crossover_rate, 0.7),
                          nil, nil, nil,
                          objectives: options[:objectives])
      end

      def nondominated?(individual, candidates)
        candidates.none? do |candidate|
          next false if individual.equal?(candidate)

          candidate.fitness.zip(individual.fitness).all? { |candidate_value, value| candidate_value <= value } &&
            candidate.fitness.zip(individual.fitness).any? { |candidate_value, value| candidate_value < value }
        end
      end
    end
  end
end
