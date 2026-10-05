# frozen_string_literal: true

require_relative '../test_helper'
require 'ai4r/genetic_algorithm/genetic_algorithm'

module Ai4r
  module GeneticAlgorithm
    class StaticObjectiveChromosome < ChromosomeBase
      attr_reader :fitness

      def initialize(label, fitness)
        super(label)
        @fitness = fitness
      end

      def self.seed
        new(:seed, [0, 0])
      end

      def self.reproduce(a, _b, _crossover_rate = 0.4)
        new(:child, a.fitness.dup)
      end

      def self.mutate(chromosome, _mutation_rate = 0.3)
        chromosome
      end
    end

    class EnumerableObjectiveChromosome < ChromosomeBase
      DOMAIN_SIZE = 8

      @seed_count = 0
      @offspring = []

      class << self
        attr_reader :offspring

        def reset!
          @seed_count = 0
          @offspring = []
        end

        def seed
          value = @seed_count % DOMAIN_SIZE
          @seed_count += 1
          new(value)
        end

        def reproduce(a, b, crossover_rate = 0.4)
          mask = 3.times.sum do |bit|
            (rand < crossover_rate ? 1 : 0) << bit
          end
          child = new((a.data & mask) | (b.data & (DOMAIN_SIZE - 1 ^ mask)))
          @offspring << child
          child
        end

        def mutate(chromosome, mutation_rate = 0.3)
          return chromosome unless rand < mutation_rate

          chromosome.data = chromosome.data ^ (1 << rand(3))
          chromosome
        end
      end

      def fitness
        [data.digits(2).count(1), data]
      end
    end

    class NsgaTwoTest < Minitest::Test
      def test_non_dominated_sort_groups_minimization_fronts_stably
        search = build_search
        values = [
          [1, 1], [1, 2], [2, 1], [2, 2]
        ]

        assert_equal [[0], [1, 2], [3]], search.nondominated_sort(values)
      end

      def test_non_dominated_sort_respects_maximization_direction
        search = build_search(objective_directions: %i[minimize maximize])
        chromosomes = objective_chromosomes(
          a: [1, 10], b: [1, 9], c: [2, 10], d: [2, 9]
        )

        selected = search.environmental_select(chromosomes.values, 4)
        ranks = chromosomes.transform_values(&:rank)

        assert_equal({ a: 0, b: 1, c: 1, d: 2 }, ranks)
        assert_equal chromosomes.values, selected
      end

      def test_truncation_prefers_fronts_then_boundary_and_crowded_individuals
        search = build_search
        chromosomes = objective_chromosomes(
          a: [0, 4], b: [1, 3], c: [2, 2], d: [3, 1], e: [4, 0]
        )

        selected = search.environmental_select(chromosomes.values, 3)

        assert_equal [chromosomes[:a], chromosomes[:e], chromosomes[:b]], selected
        assert_equal Float::INFINITY, chromosomes[:a].crowding_distance
        assert_in_delta 1.0, chromosomes[:b].crowding_distance
      end

      def test_constant_objective_and_boundary_ties_keep_combined_order
        search = build_search
        chromosomes = objective_chromosomes(
          a: [0, 1], b: [1, 0], c: [2, 2], d: [3, 1]
        )

        selected = search.environmental_select(chromosomes.values, 3)

        assert_equal [chromosomes[:a], chromosomes[:b], chromosomes[:c]], selected
        assert(chromosomes.values.all? { |chromosome| chromosome.crowding_distance.infinite? })
      end

      def test_identical_objective_vectors_keep_combined_order
        search = build_search
        chromosomes = objective_chromosomes(
          a: [1, 1], b: [1, 1], c: [1, 1], d: [2, 2]
        )

        selected = search.environmental_select(chromosomes.values, 2)

        assert_equal [chromosomes[:a], chromosomes[:b]], selected
        assert_equal [0, 0, 0, 1], chromosomes.values.map(&:rank)
        assert(chromosomes.values.first(3).all? { |chromosome| chromosome.crowding_distance.infinite? })
      end

      def test_fixed_random_evolution_returns_non_dominated_elite_set
        srand(20_261_005)
        EnumerableObjectiveChromosome.reset!
        population_size = 6
        snapshots = []

        search = nil
        callback = lambda do |generation, front|
          current = search.population
          candidates = generation.zero? ? current : combined_candidates(snapshots, generation, population_size)
          front.each { |member| refute dominated?(member, candidates) }
          assert_unchanged_elites(snapshots.last, current) if snapshots.any?
          snapshots << current.map { |chromosome| [chromosome, chromosome.data] }
        end

        search = GeneticSearch.new(
          population_size, 3, EnumerableObjectiveChromosome, 0.2, 0.7,
          nil, nil, callback, objective_directions: %i[minimize maximize]
        )
        front = search.run

        assert_equal population_size, search.population.length
        assert(front.length >= 2)
        assert_equal front.sort_by(&:data), search.pareto_front.sort_by(&:data)
        assert(front.none? { |chromosome| dominated?(chromosome, search.population) })
      end

      def test_enumerable_two_objective_domain_returns_exact_pareto_front
        EnumerableObjectiveChromosome.reset!
        search = GeneticSearch.new(
          8, 0, EnumerableObjectiveChromosome, 0.0, 0.0,
          nil, nil, nil, objective_directions: %i[minimize maximize]
        )

        front = search.run

        assert_equal [0, 4, 6, 7], front.map(&:data).sort
        assert_equal front.map(&:data), search.pareto_front.map(&:data)
      end

      private

      def build_search(objective_directions: %i[minimize minimize])
        GeneticSearch.new(
          4, 1, StaticObjectiveChromosome, 0.3, 0.4,
          nil, nil, nil,
          objective_directions: objective_directions
        )
      end

      def objective_chromosomes(labels_and_values)
        labels_and_values.transform_values do |values|
          StaticObjectiveChromosome.new(values, values)
        end
      end

      def combined_candidates(snapshots, generation, population_size)
        parents = snapshots.last.map(&:first)
        offspring_start = (generation - 1) * population_size
        children = EnumerableObjectiveChromosome.offspring[offspring_start, population_size]
        parents + children
      end

      def assert_unchanged_elites(previous_snapshot, current)
        previous_snapshot.each do |chromosome, data|
          next unless current.include?(chromosome)

          assert_equal data, chromosome.data
        end
      end

      def dominated?(member, candidates)
        candidates.any? do |other|
          no_worse = other.fitness.each_with_index.all? do |value, index|
            if index.zero?
              value <= member.fitness[index]
            else
              value >= member.fitness[index]
            end
          end
          strictly_better = other.fitness.each_with_index.any? do |value, index|
            index.zero? ? value < member.fitness[index] : value > member.fitness[index]
          end
          no_worse && strictly_better
        end
      end
    end
  end
end
