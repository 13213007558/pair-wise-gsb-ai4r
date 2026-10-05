# frozen_string_literal: true

# Author::    Sergio Fierens
# License::   MPL 1.1
# Project::   ai4r
# Url::       https://github.com/SergioFierens/ai4r
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt
require_relative 'chromosome_base'
require_relative 'tsp_chromosome'

module Ai4r
  # The GeneticAlgorithm module implements the GeneticSearch class. The
  # GeneticSearch is a generic class, and can be used to solve any kind of
  # problem. The Chromosome implementation is problem specific and must conform
  # to +ChromosomeBase+.
  module GeneticAlgorithm
    #   This class is used to automatically:
    #
    #     1. Choose initial population
    #     2. Evaluate the fitness of each individual in the population
    #     3. Repeat
    #           1. Select best-ranking individuals to reproduce
    #           2. Breed new generation through crossover and mutation (genetic operations) and give birth to offspring
    #           3. Evaluate the individual fitnesses of the offspring
    #           4. Replace worst ranked part of population with offspring
    #     4. Until termination
    #
    #   If you want to customize the algorithm, you must modify any of the following classes:
    #     - Chromosome
    #     - Population
    class GeneticSearch
      attr_accessor :population, :mutation_rate, :crossover_rate, :fitness_threshold,
                    :max_stagnation, :on_generation
      attr_reader :chromosome_class, :mode, :objective_directions, :pareto_front

      # @return [Object]
      def initialize(initial_population_size, generations, chromosome_class = TspChromosome,
                     mutation_rate = 0.3, crossover_rate = 0.4,
                     fitness_threshold = nil, max_stagnation = nil, on_generation = nil,
                     objective_directions: nil, mode: nil)
        raise ArgumentError, 'initial_population_size must be a positive Integer' unless initial_population_size.is_a?(Integer) && initial_population_size.positive?

        @population_size = initial_population_size
        @max_generation = generations
        @generation = 0
        @chromosome_class = chromosome_class
        @mutation_rate = mutation_rate
        @crossover_rate = crossover_rate
        @fitness_threshold = fitness_threshold
        @max_stagnation = max_stagnation
        @on_generation = on_generation
        @mode = mode || (objective_directions ? :nsga_ii : :standard)
        @objective_directions = objective_directions
        @pareto_front = []
        validate_multiobjective_configuration
      end

      #     1. Choose initial population
      #     2. Evaluate the fitness of each individual in the population
      #     3. Repeat
      #           1. Select best-ranking individuals to reproduce
      #           2. Breed new generation through crossover and mutation (genetic operations) and give birth to offspring
      #           3. Evaluate the individual fitnesses of the offspring
      #           4. Replace worst ranked part of population with offspring
      #     4. Until termination
      #     5. Return the best chromosome
      # @return [Object]
      def run
        multiobjective? ? run_nsga_ii : run_standard
      end

      # @return [Object]
      def generate_initial_population
        @population = []
        @population_size.times do
          population << @chromosome_class.seed
        end
      end

      # Select best-ranking individuals to reproduce
      #
      # Selection is the stage of a genetic algorithm in which individual
      # genomes are chosen from a population for later breeding.
      # There are several generic selection algorithms, such as
      # tournament selection and roulette wheel selection. We implemented the
      # latest.
      #
      # Steps:
      #
      # 1. The fitness function is evaluated for each individual, providing fitness values
      # 2. The population is sorted by descending fitness values.
      # 3. The fitness values ar then normalized. (Highest fitness gets 1, lowest fitness gets 0). The normalized value is stored in the "normalized_fitness" attribute of the chromosomes.
      # 4. A random number R is chosen. R is between 0 and the accumulated normalized value (all the normalized fitness values added togheter).
      # 5. The selected individual is the first one whose accumulated normalized value (its is normalized value plus the normalized values of the chromosomes prior it) greater than R.
      # 6. We repeat steps 4 and 5, 2/3 times the population size.
      # @return [Object]
      def selection
        @population.sort! { |a, b| b.fitness <=> a.fitness }
        best_fitness = @population[0].fitness
        worst_fitness = @population.last.fitness
        acum_fitness = 0
        if (best_fitness - worst_fitness).positive?
          @population.each do |chromosome|
            chromosome.normalized_fitness = (chromosome.fitness - worst_fitness) / (best_fitness - worst_fitness)
            acum_fitness += chromosome.normalized_fitness
          end
        else
          @population.each { |chromosome| chromosome.normalized_fitness = 1 }
        end
        selected_to_breed = []
        ((2 * @population_size) / 3).times do
          selected_to_breed << select_random_individual(acum_fitness)
        end
        selected_to_breed
      end

      # We combine each pair of selected chromosome using the method
      # Chromosome.reproduce
      #
      # The reproduction will also call the Chromosome.mutate method with
      # each member of the population. You should implement Chromosome.mutate
      # to only change (mutate) randomly. E.g. You could effectivly change the
      # chromosome only if
      #     rand < ((1 - chromosome.normalized_fitness) * 0.4)
      # @param selected_to_breed [Object]
      # @return [Object]
      def reproduction(selected_to_breed)
        offsprings = []
        0.upto((selected_to_breed.length / 2) - 1) do |i|
          offsprings << @chromosome_class.reproduce(
            selected_to_breed[2 * i],
            selected_to_breed[(2 * i) + 1],
            @crossover_rate
          )
        end
        @population.each do |individual|
          @chromosome_class.mutate(individual, @mutation_rate)
        end
        offsprings
      end

      # Replace worst ranked part of population with offspring
      # @param offsprings [Object]
      # @return [Object]
      def replace_worst_ranked(offsprings)
        size = offsprings.length
        @population = @population[0..(-size - 1)] + offsprings
      end

      # Select the best chromosome in the population
      # @return [Object]
      def best_chromosome
        the_best = @population[0]
        @population.each do |chromosome|
          the_best = chromosome if chromosome.fitness > the_best.fitness
        end
        the_best
      end

      # NSGA-II environmental selection. Parents and offspring are combined,
      # separated into non-dominated fronts, and truncated using crowding
      # distance. Ties retain the candidates' original combined order.
      def environmental_select(candidates, population_size = @population_size)
        raise ArgumentError, 'candidates must not be empty' if candidates.empty?

        values = candidates.map { |chromosome| normalized_objective_values(chromosome) }
        fronts = nondominated_sort(values)
        distances = crowding_distances(fronts, values)
        assign_selection_metadata(candidates, fronts, distances)

        selected = []
        fronts.each do |front|
          ordered_front = front.sort_by { |index| [-distances[index], index] }
          if selected.length + ordered_front.length <= population_size
            selected.concat(front)
          else
            selected.concat(ordered_front.first(population_size - selected.length))
            break
          end

          break if selected.length == population_size
        end

        selected.map { |index| candidates[index] }
      end

      # Classic fast non-dominated sorting. Front and index order remain
      # stable for candidates with identical objective vectors.
      def nondominated_sort(values)
        domination_counts = Array.new(values.length, 0)
        dominated_sets = Array.new(values.length) { [] }

        values.each_with_index do |current, index|
          ((index + 1)...values.length).each do |other_index|
            other = values[other_index]
            if dominates?(current, other)
              dominated_sets[index] << other_index
              domination_counts[other_index] += 1
            elsif dominates?(other, current)
              dominated_sets[other_index] << index
              domination_counts[index] += 1
            end
          end
        end

        fronts = []
        current_front = values.each_index.select { |index| domination_counts[index].zero? }

        until current_front.empty?
          fronts << current_front
          next_front = []
          current_front.each do |index|
            dominated_sets[index].each do |dominated_index|
              domination_counts[dominated_index] -= 1
              next_front << dominated_index if domination_counts[dominated_index].zero?
            end
          end
          current_front = next_front.sort
        end

        fronts
      end

      private

      # @param acum_fitness [Object]
      # @return [Object]
      def select_random_individual(acum_fitness)
        select_random_target = acum_fitness * rand
        local_acum = 0
        @population.each do |chromosome|
          local_acum += chromosome.normalized_fitness
          return chromosome if local_acum >= select_random_target
        end
      end

      def multiobjective?
        @mode == :nsga_ii
      end

      def run_standard
        generate_initial_population
        best = best_chromosome
        best_fitness = best.fitness
        stagnation = 0
        @on_generation&.call(@generation, best_fitness)

        @max_generation.times do
          @generation += 1
          selected_to_breed = selection
          offsprings = reproduction(selected_to_breed)
          replace_worst_ranked(offsprings)
          current_best = best_chromosome
          if current_best.fitness > best_fitness
            best_fitness = current_best.fitness
            best = current_best
            stagnation = 0
          else
            stagnation += 1
          end
          @on_generation&.call(@generation, best_fitness)
          break if (@fitness_threshold && best_fitness >= @fitness_threshold) ||
                   (@max_stagnation && stagnation >= @max_stagnation)
        end

        best
      end

      def run_nsga_ii
        generate_initial_population
        evaluate_population(@population)
        update_pareto_front
        @on_generation&.call(@generation, @pareto_front)

        @max_generation.times do
          @generation += 1
          parents = nsga_parent_selection
          offspring = nsga_reproduction(parents)
          @population = environmental_select(@population + offspring)
          update_pareto_front
          @on_generation&.call(@generation, @pareto_front)
        end

        @pareto_front
      end

      def nsga_parent_selection
        Array.new(@population_size) do
          first_index = rand(@population.length)
          second_index = rand(@population.length)
          tournament_winner(@population[first_index], first_index,
                            @population[second_index], second_index)
        end
      end

      def tournament_winner(first, first_index, second, second_index)
        return first if first.rank < second.rank
        return second if second.rank < first.rank
        return first if first.crowding_distance > second.crowding_distance
        return second if second.crowding_distance > first.crowding_distance

        first_index <= second_index ? first : second
      end

      def nsga_reproduction(parents)
        offspring = []
        pair_count = (@population_size.to_f / 2).ceil

        pair_count.times do |pair_index|
          first_parent = parents[pair_index * 2]
          second_parent = parents[(pair_index * 2) + 1] || first_parent
          offspring << @chromosome_class.reproduce(first_parent, second_parent, @crossover_rate)
          break if offspring.length == @population_size

          offspring << @chromosome_class.reproduce(second_parent, first_parent, @crossover_rate)
        end

        offspring.first(@population_size).each do |child|
          @chromosome_class.mutate(child, @mutation_rate)
        end
      end

      def evaluate_population(chromosomes)
        values = chromosomes.map { |chromosome| normalized_objective_values(chromosome) }
        fronts = nondominated_sort(values)
        distances = crowding_distances(fronts, values)
        assign_selection_metadata(chromosomes, fronts, distances)
        fronts
      end

      def update_pareto_front
        @pareto_front = @population.select { |chromosome| chromosome.rank.zero? }
      end

      def normalized_objective_values(chromosome)
        values = chromosome.fitness
        raise ArgumentError, "fitness must return an Array of #{@objective_directions.length} objectives" unless values.is_a?(Array) && values.length == @objective_directions.length

        values.each_with_index.map do |value, index|
          raise ArgumentError, "objective #{index} must be Numeric" unless value.is_a?(Numeric)

          @objective_directions[index] == :maximize ? -value : value
        end
      end

      def dominates?(values, other_values)
        no_worse = values.each_with_index.all? { |value, index| value <= other_values[index] }
        strictly_better = values.each_with_index.any? { |value, index| value < other_values[index] }
        no_worse && strictly_better
      end

      def crowding_distances(fronts, values)
        distances = Array.new(values.length, 0.0)
        fronts.each do |front|
          assign_front_crowding_distances(front, values, distances)
        end
        distances
      end

      def assign_front_crowding_distances(front, values, distances)
        return if front.empty?

        if front.length <= 2
          front.each { |index| distances[index] = Float::INFINITY }
          return
        end

        values.first.length.times do |objective|
          ordered = front.sort_by { |index| [values[index][objective], index] }
          range = values[ordered.last][objective] - values[ordered.first][objective]

          if range.zero?
            ordered.each { |index| distances[index] = Float::INFINITY }
            next
          end

          minimum = values[ordered.first][objective]
          maximum = values[ordered.last][objective]
          distances[ordered.first] = Float::INFINITY
          distances[ordered.last] = Float::INFINITY
          ordered.each do |index|
            distances[index] = Float::INFINITY if [minimum, maximum].include?(values[index][objective])
          end

          (1...(ordered.length - 1)).each do |position|
            left_index = ordered[position - 1]
            index = ordered[position]
            right_index = ordered[position + 1]
            next if distances[index].infinite?

            distances[index] += (values[right_index][objective] - values[left_index][objective]).fdiv(range)
          end
        end
      end

      def assign_selection_metadata(chromosomes, fronts, distances)
        fronts.each_with_index do |front, rank|
          front.each do |index|
            chromosome = chromosomes[index]
            chromosome.rank = rank
            chromosome.crowding_distance = distances[index]
          end
        end
      end

      def validate_multiobjective_configuration
        valid_modes = %i[standard nsga_ii].freeze
        raise ArgumentError, "mode must be one of #{valid_modes.join(', ')}" unless valid_modes.include?(@mode)

        if @mode == :standard
          return if @objective_directions.nil?

          raise ArgumentError, 'objective_directions requires mode: :nsga_ii'
        end

        raise ArgumentError, 'objective_directions must contain at least one objective' if @objective_directions.nil? || @objective_directions.empty?

        valid_directions = %i[minimize maximize].freeze
        invalid = @objective_directions - valid_directions
        return if invalid.empty?

        raise ArgumentError, "objective directions must be one of #{valid_directions.join(', ')}"
      end
    end
  end
end
