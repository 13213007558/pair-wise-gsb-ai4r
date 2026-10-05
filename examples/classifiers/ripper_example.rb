# frozen_string_literal: true

# RIPPER rule induction on a small noisy data set. The heavy item
# ["square", 31, "C", "ok"] conflicts with the other heavy items; RIPPER
# grows a condition to exclude it on the growth split and then removes that
# condition during pruning instead of memorizing the conflict.

require_relative '../../lib/ai4r/classifiers/ripper'
require_relative '../../lib/ai4r/data/data_set'

items = [
  ['square', 30, 'A', 'bad'],
  ['square', 32, 'B', 'bad'],
  ['circle', 8, 'A', 'ok'],
  ['circle', 10, 'B', 'ok'],
  ['circle', 12, 'A', 'ok'],
  ['circle', 9, 'C', 'ok'],
  ['triangle', 5, 'A', 'ok'],
  ['triangle', 7, 'B', 'ok'],
  ['square', 11, 'A', 'ok'],
  ['square', 13, 'C', 'ok'],
  ['triangle', 28, 'B', 'bad'],
  ['square', 31, 'C', 'ok']
]
labels = %w[shape weight grade label]

data = Ai4r::Data::DataSet.new(data_items: items, data_labels: labels)
classifier = Ai4r::Classifiers::Ripper.new.set_parameters(seed: 8).build(data)

puts 'Ordered rules (first match wins; otherwise the default class):'
puts classifier.get_rules
puts
puts "Default class: #{classifier.default_class}"
puts

[['square', 32, 'B'], ['square', 31, 'C'], ['circle', 8, 'A']].each do |sample|
  puts "#{sample.inspect} => #{classifier.eval(sample)}"
end

puts
puts 'Conditions removed during pruning:'
classifier.trace.each do |event|
  next unless event[:stage] == :grow

  event[:removed].each do |condition|
    label = data.data_labels[condition[:attr_index]]
    puts "  #{label} == #{condition[:value].inspect}"
  end
end
