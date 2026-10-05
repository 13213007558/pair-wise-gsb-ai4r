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
      attr_reader :chromosome_class, :objectives

      ObjectiveEntry = Struct.new(:individual, :index, :fitness, keyword_init: true)
      RankedEntry = Struct.new(:individual, :index, :fitness, :rank, :crowding_distance, keyword_init: true)

      # @return [Object]
      def initialize(initial_population_size, generations, chromosome_class = TspChromosome,
                     mutation_rate = 0.3, crossover_rate = 0.4,
                     fitness_threshold = nil, max_stagnation = nil, on_generation = nil,
                     objectives: nil)
        raise ArgumentError, 'initial_population_size must be positive' unless initial_population_size.positive?

        @population_size = initial_population_size
        @max_generation = generations
        @generation = 0
        @chromosome_class = chromosome_class
        @mutation_rate = mutation_rate
        @crossover_rate = crossover_rate
        @fitness_threshold = fitness_threshold
        @max_stagnation = max_stagnation
        @on_generation = on_generation
        @objectives = objectives.nil? ? nil : Array(objectives).map { |objective| objective.to_s.to_sym }
        @objective_directions = normalize_objectives(@objectives)
        @multi_objective = !@objective_directions.nil? && @objective_directions.any?
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
        return run_multi_objective if @multi_objective

        generate_initial_population                    # Generate initial population
        best = best_chromosome
        best_fitness = best.fitness
        stagnation = 0
        @on_generation&.call(@generation, best_fitness)
        @max_generation.times do
          @generation += 1
          selected_to_breed = selection                # Evaluates current population
          offsprings = reproduction selected_to_breed  # Generate the population for this new generation
          replace_worst_ranked offsprings
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

      # Runs NSGA-II and returns the non-dominated Pareto front.
      # @return [Array<Object>]
      def run_multi_objective
        generate_initial_population if @population.nil?
        ranked = rank_population(@population)
        @on_generation&.call(@generation, pareto_front(ranked))

        @max_generation.times do
          @generation += 1
          parents = rank_population(@population)
          offspring = reproduce_multi_objective(parents)
          combined = @population + offspring
          ranked = rank_population(combined)
          @population = environmental_selection(ranked).map(&:individual)
          ranked = rank_population(@population)
          @on_generation&.call(@generation, pareto_front(ranked))
        end

        pareto_front(rank_population(@population))
      end

      # Non-dominated fronts are returned in their original stable order.
      # @return [Array<Array<Object>>]
      def non_dominated_sort(individuals = @population)
        fast_non_dominated_sort(objective_entries(individuals)).map do |front|
          front.map(&:individual)
        end
      end

      # @return [Array<Object>]
      def pareto_front(ranked_or_individuals = nil)
        ranked = Array(ranked_or_individuals || @population)
        if ranked.first.is_a?(RankedEntry)
          return ranked.select { |entry| entry.rank.zero? }
                       .sort_by(&:index)
                       .map(&:individual)
        end

        entries = objective_entries(ranked)
        fronts = fast_non_dominated_sort(entries)
        fronts.first.map(&:individual)
      end

      # Assigns NSGA-II rank and crowding metadata in stable population order.
      # @return [Array<Struct>]
      def rank_population(individuals = @population)
        entries = objective_entries(individuals)
        fronts = fast_non_dominated_sort(entries)
        ranked = Array.new(entries.length)

        fronts.each_with_index do |front, rank|
          distances = crowding_distances(front)
          front.each do |entry|
            ranked[entry.index] = RankedEntry.new(
              individual: entry.individual,
              index: entry.index,
              fitness: entry.fitness,
              rank: rank,
              crowding_distance: distances[entry.index]
            )
          end
        end

        ranked
      end

      # Selects exactly population_size individuals using fronts and crowding.
      # @return [Array<Struct>]
      def environmental_selection(ranked)
        selected = []
        fronts = ranked.group_by(&:rank)
                       .sort
                       .map { |_, entries| entries.sort_by(&:index) }

        fronts.each do |front|
          if selected.length + front.length <= @population_size
            selected.concat(front)
          else
            needed = @population_size - selected.length
            partial = front.sort_by { |entry| [-crowding_sort_value(entry.crowding_distance), entry.index] }
            selected.concat(partial.first(needed))
            break
          end
        end

        selected
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

      private

      def normalize_objectives(objectives)
        return nil if objectives.nil?

        objectives.map do |objective|
          case objective.to_s
          when 'minimize', 'min'
            1
          when 'maximize', 'max'
            -1
          else
            raise ArgumentError, "objective must be :minimize or :maximize, got #{objective.inspect}"
          end
        end
      end

      def objective_entries(individuals)
        individuals.each_with_index.map do |individual, index|
          fitness = individual.fitness
          if @objectives
            fitness = Array(fitness)
            if fitness.length != @objectives.length
              raise ArgumentError,
                    "expected #{@objectives.length} objectives, got #{fitness.length}"
            end
          end
          ObjectiveEntry.new(individual: individual, index: index, fitness: fitness)
        end
      end

      def normalized_objective(entry, objective_index)
        value = entry.fitness[objective_index].to_f
        value * @objective_directions[objective_index]
      end

      def dominates?(first, second)
        comparisons = @objectives.each_index.map do |objective_index|
          normalized_objective(first, objective_index) <=> normalized_objective(second, objective_index)
        end
        comparisons.all? { |comparison| comparison <= 0 } && comparisons.any?(&:negative?)
      end

      def fast_non_dominated_sort(entries)
        domination_counts = Hash.new(0)
        dominated_sets = Hash.new { |hash, key| hash[key] = [] }
        fronts = [[]]

        entries.each do |candidate|
          entries.each do |other|
            next if candidate.equal?(other)

            if dominates?(candidate, other)
              dominated_sets[candidate] << other
            elsif dominates?(other, candidate)
              domination_counts[candidate] += 1
            end
          end
          fronts.first << candidate if domination_counts[candidate].zero?
        end

        current = fronts.first
        until current.empty?
          next_front = []
          current.each do |individual|
            dominated_sets[individual].each do |dominated|
              domination_counts[dominated] -= 1
              next_front << dominated if domination_counts[dominated].zero?
            end
          end
          break if next_front.empty?

          fronts << next_front
          current = next_front
        end

        fronts.map { |front| front.sort_by(&:index) }
      end

      def crowding_distances(front)
        distances = Hash.new(0.0)
        return distances if front.length <= 1

        @objectives.each_index do |objective_index|
          ordered = front.sort_by do |entry|
            [normalized_objective(entry, objective_index), entry.index]
          end
          values = ordered.map { |entry| normalized_objective(entry, objective_index) }
          min = values.first
          max = values.last
          range = max - min

          if range.zero?
            ordered.each { |entry| distances[entry.index] = [distances[entry.index], 0.0].max }
            next
          end

          ordered.each do |entry|
            value = normalized_objective(entry, objective_index)
            distances[entry.index] = Float::INFINITY if value == min || value == max
          end

          ordered.each_cons(2) do |left, right|
            left_value = normalized_objective(left, objective_index)
            right_value = normalized_objective(right, objective_index)
            next unless finite_distance?(distances[left.index]) && finite_distance?(distances[right.index])

            gap = (right_value - left_value) / range
            distances[left.index] += gap / 2.0
            distances[right.index] += gap / 2.0
          end
        end

        distances
      end

      def finite_distance?(distance)
        distance.finite?
      end

      def crowding_sort_value(distance)
        distance.infinite? ? Float::INFINITY : distance
      end

      def reproduce_multi_objective(parents)
        offspring = []
        pairs_needed = (@population_size.to_f / 2.0).ceil

        pairs_needed.times do
          first = tournament_select(parents)
          second = tournament_select(parents)
          first_parent = deep_duplicate(first)
          second_parent = deep_duplicate(second)
          first_parent.normalized_fitness = nil
          second_parent.normalized_fitness = nil
          child = @chromosome_class.reproduce(first_parent, second_parent, @crossover_rate)
          child = deep_duplicate(child)
          child.normalized_fitness = normalized_parent_fitness(parents, first)
          @chromosome_class.mutate(child, @mutation_rate)
          offspring << child
          break if offspring.length == @population_size
        end

        offspring
      end

      def tournament_select(parents)
        first = parents.sample
        second = parents.sample
        winner = if first.rank != second.rank
                   [first, second].min_by(&:rank)
                 elsif first.crowding_distance != second.crowding_distance
                   [first, second].max_by(&:crowding_distance)
                 else
                   [first, second].min_by(&:index)
                 end
        winner.individual
      end

      def normalized_parent_fitness(parents, individual)
        entry = parents.find { |ranked| ranked.individual.equal?(individual) }
        ranks = parents.map(&:rank)
        range = ranks.max - ranks.min
        return 1.0 if range.zero?

        1.0 - (entry.rank.to_f - ranks.min) / range.to_f
      end

      def deep_duplicate(individual)
        duplicate = individual.dup
        duplicate.data = deep_duplicate_value(individual.data)
        duplicate
      rescue TypeError
        individual.dup
      end

      def deep_duplicate_value(value)
        Marshal.load(Marshal.dump(value))
      rescue TypeError
        begin
          value.dup
        rescue StandardError
          value
        end
      end

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
    end
  end
end
