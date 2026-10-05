# frozen_string_literal: true

# Few-shot alert triage with Bayesian probit regression.
#
# A 0.9 alert probability can come from abundant data or from a handful of
# samples that happen to be separable. The latent response variance tells
# the two situations apart, and the predictive probability (which integrates
# parameter uncertainty) is pulled towards 0.5 when data is scarce.
#
# Pure Ruby, no dependencies beyond ai4r itself.

require 'ai4r/classifiers/bayesian_probit'
require 'ai4r/data/data_set'

# anomaly_score -> alert (1 = real alert, 0 = false alarm)
BASE_ITEMS = [
  [-2.1, 0], [-1.6, 0], [-1.2, 0], [-0.7, 0], [-0.3, 0],
  [0.4, 1], [0.9, 1], [1.3, 1], [1.8, 1], [2.4, 1]
].freeze
LABELS = %w[anomaly_score alert].freeze
PROBE = [1.1].freeze # borderline event we want to triage

def train(items, parameters = {})
  set = Ai4r::Data::DataSet.new(data_items: items, data_labels: LABELS)
  classifier = Ai4r::Classifiers::BayesianProbit.new
  classifier.set_parameters(parameters) unless parameters.empty?
  classifier.build(set)
  classifier
end

def report(name, classifier, probe)
  mean, variance = classifier.latent_response(probe)
  plug_in = 0.5 * (1.0 + Math.erf(mean / Math.sqrt(2.0)))
  template = '%<name>-22s converged=%<converged>-5s iters=%<iters>2d latent: mean=%<mean>+.3f var=%<var>7.4f | P(alert): plug-in=%<plug>.3f uncertainty-aware=%<pred>.3f'
  puts format(template, name: name, converged: classifier.converged?, iters: classifier.iterations,
                        mean: mean, var: variance, plug: plug_in, pred: classifier.predict_proba(probe))
end

puts 'Same 0.9-ish mode prediction, very different evidence:'
report('6 samples', train(BASE_ITEMS.first(4) + BASE_ITEMS.last(2)), PROBE)
report('60 samples', train(BASE_ITEMS * 6), PROBE)

puts
puts 'Repeating the same 10 observations (uncertainty vs data volume):'
[1, 2, 4, 8].each do |copies|
  report("#{copies}x batch (#{BASE_ITEMS.length * copies} rows)", train(BASE_ITEMS * copies), PROBE)
end

puts
puts 'Custom prior (feature weights ~ N(0, 1), bias ~ N(0, 25)):'
report('weakly informative', train(BASE_ITEMS, prior_variance: 1.0, bias_prior_variance: 25.0), PROBE)
