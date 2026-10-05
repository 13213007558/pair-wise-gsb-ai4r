# CART Decision Tree

`Ai4r::Classifiers::CART` is a binary classification tree in the style of
Breiman, Friedman, Olshen and Stone. It uses the same `DataSet` interface as
the other classifiers: the last column is the class.

```ruby
require 'ai4r'

labels = %w[reading model fault]
items = [
  [1.0, 'A', 'no'], [2.0, 'A', 'no'], [8.0, 'B', 'yes'], [9.0, 'B', 'yes']
]
data = Ai4r::Data::DataSet.new(data_items: items, data_labels: labels)
cart = Ai4r::Classifiers::CART.new.build(data)
cart.eval([8.5, 'B']) # => 'yes'
```

## Binary splits

Every node makes one yes/no test:

* numeric features: `feature <= threshold` goes left, otherwise right
* discrete features: membership in a trained subset goes left, otherwise right

Inspect `cart.root` for the chosen `feature_index`, `threshold` or `subset`,
`surrogates`, children and `default_direction`. `get_rules`, `to_h` and
`to_graphviz` give readable explanations of every split.

## Weights and misclassification costs

Pass one non-negative weight per example. The same weights are used for split
selection, the weighted class distribution stored at each leaf, and resubstitution
risk used while pruning.

```ruby
cart = Ai4r::Classifiers::CART.new.build(
  data,
  weights: [1, 1, 1, 10],
  costs: { 'no' => { 'yes' => 5 } }
)
cart.root.left_child.distribution # => { 'no' => 4.0 }
```

A leaf label minimizes `weight * cost(predicted, actual)`. Without `costs`,
a wrong prediction costs one and a correct prediction costs zero; rare classes
can therefore be emphasized with weights, costs, or both.

## Missing values and surrogate splits

Represent a missing sensor reading with `nil`. Training ignores a row for a
candidate primary split when that feature is missing, but still uses the row for
other features. Each internal node learns ordered surrogate splits that best
reproduce the primary direction. At prediction time:

- (1) use the primary split when its feature is present;
- (2) otherwise use the first surrogate whose feature is present;
- (3) if no surrogate is available, take the node's trained `default_direction`
  (the side carrying more observed training weight).

`cart.path_for(data)` reports each chosen condition and whether a surrogate was
used, which makes a missing-value decision easy to audit.

## Weakest-link pruning

`build` also computes `cart.pruning_sequence`, a nested sequence of subtrees
produced by weakest-link cost-complexity pruning. Each entry contains its
`alpha`, number of leaves and a snapshot tree. Select a tree by alpha:

```ruby
cart.pruning_sequence.each do |entry|
  p alpha: entry[:alpha], leaves: entry[:leaves]
end

cart.eval_with_alpha([8.5, 'B'], 0.1)
pruned_tree = cart.tree_for_alpha(0.1)
```

Alphas are non-decreasing and leaf counts are non-increasing, so a subtree
removed at one alpha never reappears at a larger alpha.

## Parameters

* `max_depth` - stop after this depth (default: unlimited)
* `min_weight` - minimum node weight required to split (default: `1`)
* `min_gain` - minimum gain required to split (default: `0`)
* `max_categorical_values` - domains at most this large use an exhaustive
  subset search; larger domains use one-vs-rest candidates (default: `12`)
* `weights` - per-example non-negative weights
* `costs` - nested hash of non-negative misclassification costs
