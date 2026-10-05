# RIPPER-Style Rule Learning

`RipperClassifier` learns an ordered decision list for noisy classification data.
Rules are evaluated from top to bottom and the first matching rule wins. When no
rule matches, the default class is returned.

## Fixed semantics

* The default class is the training majority class, unless `default_class` is set.
* The remaining classes are learned from least frequent to most frequent.
* Prediction uses the first matching rule; later rules cannot override it.
* Each class uses a deterministic stratified split of its remaining examples:
  two thirds for growing and one third for pruning. Set `random_seed` to make the
  split reproducible.

## Learning stages

1. Grow one rule with FOIL-style gain. Conditions may be:
   * discrete equality: `color == 'red'`
   * continuous thresholds: `weight <= 4.5` or `weight > 9.5`
2. Greedily delete conditions while the IREP pruning score improves on the
   pruning partition. A rule is not allowed to prune down to an empty rule,
   because an empty rule would mean the default class.
3. Accept the rule only when it reduces the approximate description length of
   the rule set plus the misclassified examples left to encode.
4. Optimize every learned rule by comparing:
   * the kept rule,
   * a replacement grown from an empty antecedent,
   * a revision grown starting from the existing antecedent.

## Noisy small sample: stopping instead of lengthening a rule

The last column is the label. `F` is the minority class and `N` is supplied as
the fixed default class.

```ruby
items = [
  ['red', 'S',  0, 'F'], ['red', 'M',  1, 'F'], ['red', 'S',  2, 'F'],
  ['red', 'M',  3, 'F'], ['red', 'L',  4, 'F'],
  ['blue', 'L', 10, 'F'], ['blue', 'M', 11, 'F'], ['blue', 'S', 12, 'F'],
  ['blue', 'M', 13, 'F'], ['blue', 'L', 14, 'F'],
  ['red', 'S',  5, 'N'], ['green', 'L',  6, 'N'], ['red', 'M',  7, 'N'],
  ['green', 'S',  8, 'N'], ['blue', 'S',  9, 'N']
]
labels = %w[color size weight label]
data = Ai4r::Data::DataSet.new(data_items: items, data_labels: labels)
classifier = Ai4r::Classifiers::RipperClassifier
             .new
             .set_parameters(random_seed: 2, default_class: 'N')
             .build(data)
puts classifier.get_rules
```

The middle rows contain conflicting labels: some red items with weights 5 and 7
are labelled `N`, while low-weight red items and high-weight blue items are `F`.
The learner keeps two compact interval-like rules rather than adding color and
size conditions to cover every positive-looking training row:

```text
RULE TABLE (first matching rule wins)
1. IF weight <= 4.5 THEN label = 'F'
2. IF weight > 9.5 THEN label = 'F'
DEFAULT label = 'N'
```

The final conditions are the two weight thresholds. They separate the clean
low-weight and high-weight positive regions. The contradictory middle region is
left to the default class, so the model does not keep adding conditions such as
`color == 'red'`, `size == 'S'`, or increasingly narrow weight conjunctions just
to memorize every noisy training row.

## Noisy small sample: a condition deleted by pruning

A second sample makes the deletion visible. The grow set contains a positive low
weight cluster, a positive high weight cluster, and a red high-weight negative
row:

```ruby
items = []
5.times { |i| items << ['red', i.even? ? 'S' : 'M', i, 'F'] }
5.times { |i| items << ['blue', i.even? ? 'M' : 'L', 10 + i, 'F'] }
5.times do |i|
  items << [%w[red green][i % 2], i.even? ? 'S' : 'L', 5 + i, 'N']
end
```

Using `random_seed: 2`, growing temporarily produces:

```ruby
weight > 4.5 AND color == 'red'
```

After evaluating that antecedent on the pruning partition, the discrete equality
condition `color == 'red'` is removed. The final rule is only:

```ruby
weight > 4.5 # => N
```

The removed equality condition was an artifact of the grow partition: it helped
separate one noisy red row there, but it was not the useful separator on the
prune partition. The continuous threshold remains because it describes the
actual high-weight region.

`rule_traces` records each stage. For example, the traces show the grown and
pruned conditions, the number of grow/prune examples, and the description length
before and after accepting each rule.

## Portable rules without `eval`

Use `export_rules` for a plain data representation that can be serialized as
JSON or passed to an operations tool. `execute_rules` interprets that table
directly; it does not generate or evaluate Ruby source code.

```ruby
exported = classifier.export_rules
prediction = Ai4r::Classifiers::RipperClassifier.execute_rules(
  exported, ['red', 'S', 7]
)
# => 'N'
```

The exported representation uses positional attributes:

```ruby
{
  version: 1,
  semantics: { evaluation: :first_match, default_class: 'N' },
  rules: [
    { conditions: [{ attribute_index: 2, operator: '<=', value: 4.5 }],
      class_value: 'F' }
  ]
}
```
