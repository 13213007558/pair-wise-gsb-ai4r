# frozen_string_literal: true

require_relative '../../lib/ai4r/classifiers/bayesian_probit'
require_relative '../../lib/ai4r/data/data_set'

items = [
  [-3.0, 0],
  [-2.0, 0],
  [1.0, 1],
  [2.0, 1]
]

model = Ai4r::Classifiers::BayesianProbit.new
model.set_parameters(slope_prior_variance: 1.0, bias_prior_variance: 100.0)
model.build(Ai4r::Data::DataSet.new(data_items: items))

puts model.optimization_status
puts "posterior mode: #{model.posterior_mode.map { |value| format('%.4f', value) }.join(', ')}"
puts "warnings: #{model.warnings.join('; ')}"

[0.0, 1.0].each do |point|
  mean, variance = model.latent_linear_response([point])
  probability = model.positive_class_probability([point])
  plugin = 0.5 * Math.erfc(-mean / Math.sqrt(2.0))
  puts "x=#{point} mean=#{format('%.4f', mean)} variance=#{format('%.4f', variance)} " \
       "p(y=1)=#{format('%.4f', probability)} mode-only=#{format('%.4f', plugin)}"
end

repeated = items.flat_map { |item| Array.new(8, item.dup) }
repeated_model = Ai4r::Classifiers::BayesianProbit.new
repeated_model.build(Ai4r::Data::DataSet.new(data_items: repeated))
_, initial_variance = model.latent_linear_response([1.0])
_, repeated_variance = repeated_model.latent_linear_response([1.0])
puts "variance at x=1: #{format('%.4f', initial_variance)} -> #{format('%.4f', repeated_variance)}"
