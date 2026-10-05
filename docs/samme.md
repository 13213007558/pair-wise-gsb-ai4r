# SAMME (Multi-class AdaBoost)

`Ai4r::Classifiers::Samme` combines weak classifiers into a single multi-class
ensemble using the SAMME algorithm (Stagewise Additive Modeling using a
Multiclass Exponential loss). Each round trains one multi-class weak learner
on reweighted samples; the problem is never split into one-vs-rest binary
tasks. Class labels can be any Ruby object, including strings.

## Parameters

* `n_estimators` – maximum number of boosting rounds. Default is `10`.
* `learning_rate` – shrinkage applied to each learner weight. Default is `1.0`.
* `learner_factory` – callable returning a fresh weak learner per round. It is
  called as `factory.call` or, when its arity is not zero, `factory.call(rng)`
  with the round RNG. The learner must respond to
  `build(data_set, sample_weights)` and `eval(data)`.
* `random_seed` – seed for the RNG handed to the factory, so stochastic
  learners reproduce identical round records.

## Round records

Every trained round is stored in `rounds` as a struct with:

* `learner` – the weak learner trained that round.
* `alpha` – the learner weight.
* `error` – the weighted training error of the learner.
* `classes` – the class mapping (index to label) for that round.
* `sample_weights` – the normalized sample weights after the round's update.

Use `eval(data, n)` and `scores(data, n)` to predict with only the first `n`
rounds; `scores` returns the summed alpha votes per class label.

## Edge cases

* Zero weighted error: the round is recorded with alpha `MAX_ALPHA` and
  training stops.
* Error at or above the random guessing bound (`1 - 1/K`): training stops; if
  it happens on the first round, the learner is kept with alpha `1.0`.
* A single class in the data: no rounds are trained and `eval` always returns
  that class.

Sample weights are renormalized after every round and fall back to a uniform
distribution on numerical failure, so they never underflow to all zeros.

## Example

```ruby
require 'ai4r/classifiers/samme'
require 'ai4r/data/data_set'

set = Ai4r::Data::DataSet.new(data_items: items, data_labels: labels)
samme = Ai4r::Classifiers::Samme.new.set_parameters(
  n_estimators: 20,
  learner_factory: -> { MyWeightedStump.new }
).build(set)

samme.eval([1.0, 2.0])        # prediction using all rounds
samme.eval([1.0, 2.0], 5)     # prediction using the first 5 rounds
samme.scores([1.0, 2.0])      # per-class SAMME scores
```

This algorithm does not produce interpretable rules.
