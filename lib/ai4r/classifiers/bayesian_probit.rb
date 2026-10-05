# frozen_string_literal: true

require_relative '../data/data_set'
require_relative 'classifier'

module Ai4r
  module Classifiers
    # Binary Bayesian probit classifier using a Gaussian parameter prior and
    # a Laplace approximation to the parameter posterior.
    class BayesianProbit < Classifier
      NumericFailure = Class.new(RuntimeError)

      MAX_FINITE_VALUE = 1.0e150
      SQRT_TWO_PI = Math.sqrt(2.0 * Math::PI)
      LOG_SQRT_TWO_PI = Math.log(SQRT_TWO_PI)

      attr_reader :posterior_mode, :posterior_covariance, :prior_mean,
                  :prior_precision, :feature_count, :classes, :rank,
                  :converged, :iterations, :optimization_status, :warnings

      parameters_info slope_prior_mean: 'Prior mean for feature coefficients; a number or Array.',
                      slope_prior_variance: 'Positive scalar, Array of variances, or covariance matrix for slopes.',
                      bias_prior_mean: 'Prior mean for the automatically appended bias. Default 0.0.',
                      bias_prior_variance: 'Prior variance for the automatically appended bias. Default 100.0.',
                      max_iterations: 'Maximum Newton iterations. Default 50.',
                      tolerance: 'Infinity-norm Newton convergence tolerance. Default 1e-8.',
                      line_search_shrink: 'Backtracking line-search shrink factor. Default 0.5.',
                      integration_tolerance: 'Adaptive Simpson predictive integration tolerance. Default 1e-10.',
                      integration_stdev: 'Integration half-width in latent-response standard deviations. Default 8.0.',
                      integration_depth: 'Maximum adaptive Simpson recursion depth. Default 20.'

      def initialize
        super
        @slope_prior_mean = 0.0
        @slope_prior_variance = 1.0
        @bias_prior_mean = 0.0
        @bias_prior_variance = 100.0
        @max_iterations = 50
        @tolerance = 1.0e-8
        @line_search_shrink = 0.5
        @integration_tolerance = 1.0e-10
        @integration_stdev = 8.0
        @integration_depth = 20
        reset_fit
      end

      def build(data_set)
        raise 'Error instance must be passed' unless data_set.is_a?(Ai4r::Data::DataSet)
        raise 'Data should not be empty' if data_set.data_items.empty?

        features, targets, classes = extract_training_data(data_set)
        @classes = classes
        @feature_count = features.first.length
        @prior_mean = build_prior_mean
        @prior_precision = build_prior_precision
        @warnings = []

        design = features.map { |row| row + [1.0] }
        check_matrix_scale(design)
        @rank = matrix_rank(design)
        if @rank < design.first.length
          @warnings << 'Design matrix is rank deficient; one or more features are collinear or constant. The Gaussian prior keeps the posterior identifiable, but inspect posterior_covariance.'
        end
        if separable?(design, targets)
          @warnings << 'Training data are completely or quasi-completely separated. The unregularized likelihood has no finite mode; the result depends on the Gaussian prior.'
        end

        fit_mode(design, targets)
        self
      end

      def eval(data)
        positive_class_probability(data) >= 0.5 ? @classes.last : @classes.first
      end

      def get_probability_map(data)
        probability = positive_class_probability(data)
        { @classes.first => 1.0 - probability, @classes.last => probability }
      end

      def latent_linear_response(data)
        ensure_trained
        augmented = validate_features(data) + [1.0]
        mean = dot(augmented, @posterior_mode)
        variance = quadratic(augmented, @posterior_covariance)
        ensure_finite(mean, 'latent response mean')
        ensure_finite(variance, 'latent response variance')
        raise NumericFailure, 'Latent response variance is negative' if variance.negative?

        [mean, variance]
      end

      def positive_class_probability(data)
        mean, variance = latent_linear_response(data)
        integrate_normal_probit(mean, Math.sqrt(variance))
      end

      def get_rules
        'BayesianProbit does not support rule extraction.'
      end

      private

      def reset_fit
        @posterior_mode = nil
        @posterior_covariance = nil
        @prior_mean = nil
        @prior_precision = nil
        @feature_count = nil
        @classes = nil
        @rank = nil
        @converged = false
        @iterations = 0
        @optimization_status = 'not trained'
        @warnings = []
      end

      def extract_training_data(data_set)
        item_length = data_set.data_items.first.length
        raise ArgumentError, 'Each training item must include features and a class label' if item_length < 2

        labels = data_set.data_items.map(&:last).uniq.sort
        raise ArgumentError, 'BayesianProbit requires exactly two classes' unless labels.length == 2

        features = []
        targets = []
        data_set.data_items.each do |item|
          raise ArgumentError, 'All training items must have the same number of fields' unless item.length == item_length

          row = item[0...-1].map { |value| to_float(value, 'feature') }
          check_vector_scale(row)
          raise ArgumentError, "Unexpected class #{item.last.inspect}" unless labels.include?(item.last)

          features << row
          targets << (item.last == labels.last ? 1.0 : 0.0)
        end
        [features, targets, labels]
      end

      def build_prior_mean
        slope_means = expand_vector(@slope_prior_mean, @feature_count, 'slope_prior_mean')
        bias_mean = to_float(@bias_prior_mean, 'bias_prior_mean')
        raise ArgumentError, 'prior means must be finite' unless slope_means.all?(&:finite?) && bias_mean.finite?

        slope_means + [bias_mean]
      end

      def build_prior_precision
        slope_precision = slope_prior_precision
        bias_variance = to_float(@bias_prior_variance, 'bias_prior_variance')
        raise ArgumentError, 'bias_prior_variance must be positive and finite' unless bias_variance.positive? && bias_variance.finite?

        count = @feature_count + 1
        precision = Array.new(count) { Array.new(count, 0.0) }
        slope_precision.each_with_index do |row, i|
          row.each_with_index { |value, j| precision[i][j] = value }
        end
        precision[@feature_count][@feature_count] = 1.0 / bias_variance
        precision
      end

      def slope_prior_precision
        case @slope_prior_variance
        when Numeric
          variance = to_float(@slope_prior_variance, 'slope_prior_variance')
          raise ArgumentError, 'slope_prior_variance must be positive and finite' unless variance.positive? && variance.finite?

          Array.new(@feature_count) do |i|
            Array.new(@feature_count) { |j| i == j ? 1.0 / variance : 0.0 }
          end
        when Array
          covariance = coerce_matrix(@slope_prior_variance, @feature_count, @feature_count, 'slope_prior_variance')
          inverse(covariance)
        else
          raise ArgumentError, 'slope_prior_variance must be a number, variance Array, or covariance matrix'
        end
      end

      def fit_mode(design, targets)
        beta = @prior_mean.dup
        best_objective = nil
        converged = false

        @max_iterations.times do |iteration|
          scores = design.map { |row| dot(row, beta) }
          check_vector_scale(scores)
          weights, signals = likelihood_terms(scores, targets)
          gradient = score_gradient(design, signals, beta)
          hessian = observed_hessian(design, weights)
          step = solve(hessian, gradient)
          objective = negative_log_posterior(scores, targets, beta)
          candidate, candidate_objective = line_search(design, targets, beta, step, objective, gradient)
          step_size = vector_subtract(candidate, beta).map(&:abs).max
          beta = candidate
          best_objective = candidate_objective
          @iterations = iteration + 1

          next unless step_size <= @tolerance

          finalize_fit(design, targets, beta)
          @converged = true
          converged = true
          @optimization_status = "converged after #{@iterations} Newton iterations"
          break
        end

        unless converged
          finalize_fit(design, targets, beta)
          @converged = false
          @optimization_status = "maximum iterations (#{@max_iterations}) reached without meeting tolerance #{@tolerance}; objective #{best_objective}"
        end
      rescue NumericFailure => e
        @posterior_mode = beta
        @posterior_covariance = nil
        @converged = false
        @optimization_status = "numerical failure: #{e.message}"
        raise
      end

      def finalize_fit(design, targets, beta)
        scores = design.map { |row| dot(row, beta) }
        check_vector_scale(scores)
        weights, = likelihood_terms(scores, targets)
        @posterior_mode = beta
        @posterior_covariance = inverse(observed_hessian(design, weights))
        ensure_finite_vector(@posterior_mode, 'posterior mode')
        @posterior_covariance.each { |row| ensure_finite_vector(row, 'posterior covariance') }
      end

      def line_search(design, targets, beta, step, objective, gradient)
        directional = dot(gradient, step)
        step_size = 1.0
        60.times do
          candidate = beta.zip(step).map { |value, delta| value + (step_size * delta) }
          scores = design.map { |row| dot(row, candidate) }
          check_vector_scale(scores)
          candidate_objective = negative_log_posterior(scores, targets, candidate)
          return [candidate, candidate_objective] if candidate_objective <= objective - (1.0e-4 * step_size * directional)

          step_size *= @line_search_shrink
        end
        raise NumericFailure, 'backtracking line search failed to improve the regularized objective'
      end

      def likelihood_terms(scores, targets)
        weights = []
        signals = []
        scores.zip(targets) do |score, target|
          signal = target > 0.5 ? inverse_mills_upper(score) : -inverse_mills_lower(score)
          weight = signal * (signal - score)
          raise NumericFailure, 'probit likelihood weight became non-finite or excessively large; rescale input features' unless weight.finite? && weight >= 0.0 && weight <= 1.0e10

          weights << weight
          signals << signal
        end
        [weights, signals]
      end

      def score_gradient(design, signals, beta)
        gradient = Array.new(beta.length, 0.0)
        design.each_with_index do |row, i|
          row.each_with_index { |value, j| gradient[j] += value * signals[i] }
        end
        difference = vector_subtract(beta, @prior_mean)
        beta.length.times { |j| gradient[j] += dot(@prior_precision[j], difference) }
        gradient
      end

      def observed_hessian(design, weights)
        count = design.first.length
        hessian = @prior_precision.map(&:dup)
        design.each_with_index do |row, i|
          count.times do |j|
            count.times { |k| hessian[j][k] += weights[i] * row[j] * row[k] }
          end
        end
        hessian
      end

      def negative_log_posterior(scores, targets, beta)
        likelihood = scores.zip(targets).sum(0.0) do |score, target|
          target > 0.5 ? -normal_log_cdf(score) : -normal_log_cdf(-score)
        end
        difference = vector_subtract(beta, @prior_mean)
        likelihood + (0.5 * quadratic(difference, @prior_precision))
      end

      def normal_cdf(value)
        0.5 * Math.erfc(-value / Math.sqrt(2.0))
      end

      def normal_pdf(value)
        Math.exp(-0.5 * value * value) / SQRT_TWO_PI
      end

      def normal_log_cdf(value)
        if value >= -37.0
          cdf = normal_cdf(value)
          raise NumericFailure, 'normal CDF underflowed before asymptotic range' unless cdf.positive?

          Math.log(cdf)
        else
          normal_log_lower_tail(value.abs)
        end
      end

      def normal_log_lower_tail(value)
        inverse = inverse_mills_lower(-value)
        (-0.5 * value * value) - LOG_SQRT_TWO_PI - Math.log(inverse)
      end

      def inverse_mills_upper(value)
        if value <= -37.0
          -value
        elsif value < 37.0
          tail = 0.5 * Math.erfc(value / Math.sqrt(2.0))
          raise NumericFailure, 'upper normal tail underflowed unexpectedly' unless tail.positive?

          density = normal_pdf(value)
          raise NumericFailure, 'normal density underflowed unexpectedly' unless density.positive?

          density / tail
        else
          value + (1.0 / value) - (1.0 / value**3) + (3.0 / value**5) -
            (15.0 / value**7) + (105.0 / value**9) - (945.0 / value**11)
        end
      end

      def inverse_mills_lower(value)
        inverse_mills_upper(-value)
      end

      def integrate_normal_probit(mean, standard_deviation)
        raise NumericFailure, 'predictive latent response is NaN' unless standard_deviation.finite? || mean.finite? || mean.infinite?
        return 0.0 if mean == -Float::INFINITY
        return 1.0 if mean == Float::INFINITY
        return normal_cdf(mean) if standard_deviation.zero?
        raise NumericFailure, 'predictive response variance must be finite and non-negative' unless standard_deviation.positive? && standard_deviation.finite? && mean.finite?

        lower = -@integration_stdev
        upper = @integration_stdev
        left_middle = simpson_point(mean, standard_deviation, lower, 0.0)
        right_middle = simpson_point(mean, standard_deviation, 0.0, upper)
        adaptive_simpson(mean, standard_deviation, lower, 0.0, left_middle, @integration_depth) +
          adaptive_simpson(mean, standard_deviation, 0.0, upper, right_middle, @integration_depth)
      end

      def adaptive_simpson(mean, standard_deviation, lower, upper, whole, depth)
        middle = (lower + upper) / 2.0
        left = simpson_point(mean, standard_deviation, lower, middle)
        right = simpson_point(mean, standard_deviation, middle, upper)
        refined = left + right
        error = (refined - whole).abs / 15.0
        allowed_error = @integration_tolerance * ((upper - lower) / (2.0 * @integration_stdev))
        return refined + ((refined - whole) / 15.0) if depth.zero? || error <= allowed_error

        adaptive_simpson(mean, standard_deviation, lower, middle, left, depth - 1) +
          adaptive_simpson(mean, standard_deviation, middle, upper, right, depth - 1)
      end

      def simpson_point(mean, standard_deviation, lower, upper)
        middle = (lower + upper) / 2.0
        (upper - lower) / 6.0 *
          (integration_value(mean, standard_deviation, lower) +
           (4.0 * integration_value(mean, standard_deviation, middle)) +
           integration_value(mean, standard_deviation, upper))
      end

      def integration_value(mean, standard_deviation, point)
        normal_pdf(point) * normal_cdf(mean + (standard_deviation * point))
      end

      def ensure_trained
        raise 'Model not trained' unless @posterior_mode
      end

      def validate_features(data)
        raise ArgumentError, 'Data must be an Array' unless data.is_a?(Array)
        raise ArgumentError, "Wrong number of inputs. Expected: #{@feature_count}, received: #{data.length}." unless data.length == @feature_count

        row = data.map { |value| to_float(value, 'feature') }
        check_vector_scale(row)
        row
      end

      def to_float(value, name)
        if value.is_a?(Numeric)
          float = value.to_f
          raise ArgumentError, "#{name} must be finite" unless float.finite?

          return float
        end

        raise ArgumentError, "#{name} must be numeric, received #{value.inspect}"
      end

      def expand_vector(value, size, name)
        if value.is_a?(Array)
          raise ArgumentError, "#{name} must have length #{size}" unless value.length == size && value.all?(Numeric)

          value.map(&:to_f)
        elsif value.is_a?(Numeric)
          Array.new(size, value)
        else
          raise ArgumentError, "#{name} must be a number or Array"
        end
      end

      def coerce_matrix(value, rows, columns, name)
        raise ArgumentError, "#{name} must be a #{rows}x#{columns} matrix" unless value.is_a?(Array) && value.length == rows && value.all?(Array)
        raise ArgumentError, "#{name} must contain #{columns} numeric values per row" unless value.all? { |row| row.length == columns && row.all?(Numeric) }

        matrix = value.map { |row| row.map(&:to_f) }
        rows.times do |i|
          columns.times do |j|
            finite = matrix[i][j].finite? && matrix[j][i].finite?
            symmetric = (matrix[i][j] - matrix[j][i]).abs <= 1.0e-10 * (1.0 + matrix[i][j].abs)
            raise ArgumentError, "#{name} must be finite and symmetric" unless finite && symmetric
          end
        end
        matrix
      end

      def dot(left, right)
        left.each_with_index.sum(0.0) { |value, i| value * right[i] }
      end

      def quadratic(vector, matrix)
        vector.each_with_index.sum(0.0) do |value, i|
          value * dot(matrix[i], vector)
        end
      end

      def vector_subtract(left, right)
        left.zip(right).map { |a, b| a - b }
      end

      def add_in_place(left, right, scale = 1.0)
        left.length.times { |i| left[i] += scale * right[i] }
        left
      end

      def cholesky(matrix)
        size = matrix.length
        factor = Array.new(size) { Array.new(size, 0.0) }
        size.times do |i|
          size.times do |j|
            next if j > i

            sum = (0...j).sum(0.0) { |k| factor[i][k] * factor[j][k] }
            factor[i][j] = if i == j
                             diagonal = matrix[i][i] - sum
                             raise NumericFailure, 'matrix is not positive definite' unless diagonal.positive? && diagonal.finite?

                             Math.sqrt(diagonal)
                           else
                             (matrix[i][j] - sum) / factor[j][j]
                           end
          end
        end
        factor
      end

      def solve(matrix, vector)
        factor = cholesky(matrix)
        forward_solve(factor, vector)
      end

      def forward_solve(lower, vector)
        size = lower.length
        y = Array.new(size, 0.0)
        size.times do |i|
          sum = (0...i).sum(0.0) { |j| lower[i][j] * y[j] }
          y[i] = (vector[i] - sum) / lower[i][i]
        end
        backward_solve(lower, y)
      end

      def backward_solve(lower, vector)
        size = lower.length
        solution = Array.new(size, 0.0)
        (size - 1).downto(0) do |i|
          sum = ((i + 1)...size).sum(0.0) { |j| lower[j][i] * solution[j] }
          solution[i] = (vector[i] - sum) / lower[i][i]
        end
        solution
      end

      def inverse(matrix)
        size = matrix.length
        factor = cholesky(matrix)
        Array.new(size) do |column|
          unit = Array.new(size, 0.0)
          unit[column] = 1.0
          forward_solve(factor, unit)
        end.transpose
      end

      def check_matrix_scale(matrix)
        matrix.each { |row| check_vector_scale(row) }
      end

      def check_vector_scale(vector)
        vector.each do |value|
          unless value.is_a?(Numeric) && value.to_f.finite? && value.abs <= MAX_FINITE_VALUE
            raise ArgumentError, 'all numeric values must be finite and no larger in magnitude than 1.0e150; rescale the data'
          end
        end
      end

      def ensure_finite(value, name)
        raise NumericFailure, "#{name} is not finite" unless value.finite?
      end

      def ensure_finite_vector(vector, name)
        vector.each { |value| ensure_finite(value, name) }
      end

      def matrix_rank(matrix)
        copy = matrix.map(&:dup)
        columns = copy.first.length
        scale = 1.0
        matrix.each { |row| row.each { |value| scale = [scale, value.abs].max } }
        tolerance = 1.0e-12 * scale
        rank = 0
        columns.times do |column|
          pivot_row = rank.upto(copy.length - 1).find { |row| copy[row][column].abs > tolerance }
          next unless pivot_row

          copy[rank], copy[pivot_row] = copy[pivot_row], copy[rank]
          pivot = copy[rank][column]
          (rank + 1).upto(copy.length - 1) do |row|
            factor = copy[row][column] / pivot
            column.upto(columns - 1) { |j| copy[row][j] -= factor * copy[rank][j] }
          end
          rank += 1
        end
        rank
      end

      def separable?(design, targets)
        weights = Array.new(design.first.length, 0.0)
        result = 10_000.times do
          changed = false
          design.zip(targets) do |row, target|
            score = dot(row, weights)
            misclassified_positive = target > 0.5 && score <= 0.0
            misclassified_negative = target < 0.5 && score >= 0.0
            next unless misclassified_positive || misclassified_negative

            direction = target > 0.5 ? 1.0 : -1.0
            add_in_place(weights, row, direction)
            changed = true
          end
          break true unless changed
        end
        result == true
      end
    end
  end
end
