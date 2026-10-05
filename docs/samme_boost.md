# SAMME Boosting

`Ai4r::Classifiers::SAMMEBoost` combines weighted weak learners with the discrete SAMME multiclass algorithm. It trains each learner on the complete label set; it does not create one-vs-rest binary problems.

## Parameters

* `n_estimators` – maximum number of boosting rounds. Default is `10`.
* `random_seed` – integer seed or `Random` instance supplied to the weak learner factory.
* `weak_learner_factory` – callable returning a learner whose `build(data_set, sample_weights)` method accepts sample weights and whose `eval(data)` returns a training label.

## Inspection

Each entry in `rounds` contains:

* `classes` and `class_mapping` for that round
* `learner`, `learner_weight`, and `weighted_error`
* `sample_weights` before the round and `updated_sample_weights` afterward
* `status`, one of `:added`, `:zero_error`, `:random_guess`, or `:worse_than_random`

Use `eval(data, round: n)` and `class_scores(data, round: n)` to inspect a cumulative prediction through round `n`. A zero-error round gets infinite learner weight and stops training. A random-guess or worse learner is not added to scores and also stops training. Training raises `ArgumentError` when the data contains only one class.

## Example

```ruby
require 'ai4r/classifiers/samme_boost'
require 'ai4r/data/data_set'

factory = ->(rng) { MyWeightedWeakLearner.new(rng) }
set = Ai4r::Data::DataSet.new(data_items: [[0, 'a'], [1, 'b'], [2, 'c']])
boost = Ai4r::Classifiers::SAMMEBoost
          .new
          .set_parameters(n_estimators: 5, weak_learner_factory: factory, random_seed: 42)
          .build(set)

boost.eval([0])
boost.class_scores([0])
boost.rounds.first.weighted_error
```
