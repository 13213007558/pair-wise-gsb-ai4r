# RIPPER Rule Induction

RIPPER (Repeated Incremental Pruning to Produce Error Reduction, Cohen 1995)
induces an ordered list of if-then rules and is designed for noisy data,
where [PRISM](prism.md) tends to grow long rules just to memorize conflicting
training items.

## How it works

1. Classes are ordered by frequency. The minority classes are learned first;
   the most frequent class becomes the default class and never appears in a
   rule head. Frequency ties follow first appearance in the training data.
2. Training items are split into a growth set and a prune set
   (prune_ratio, deterministic with seed).
3. For each minority class, rules are grown on the growth set with FOIL
   information gain, then reduced-error pruned on the prune set.
4. A pruned rule is kept only while it decreases the description length of
   the rules plus the remaining errors, so noise does not justify extra
   conditions.
5. An optimization pass revisits every rule and chooses between the original
   rule, a replacement grown from scratch, and a revision grown from the
   original rule's conditions.

Conditions are either discrete equality tests (attr == value) or numeric
threshold tests (attr <= value, attr >= value). Thresholds are midpoints
between adjacent distinct values.

## Prediction semantics

The rules are checked in their stored order and the first rule whose
conditions all match provides the answer. If no rule matches, the default
class is returned. Prediction traverses the plain rule data structure and
never calls Kernel#eval.

## Inspecting and exporting rules

to_rules returns plain data: an ordered array of class/conditions hashes and
the default_class. Running the conditions in that order reproduces eval
exactly. get_rules renders the same rule set as an executable Ruby
if/elsif/else chain for inspection.

## Worked example with conflicting labels

See examples/classifiers/ripper_example.rb. On the growth split the single
heavy item square/31/ok conflicts with the heavy bad items. Growing first
adds weight >= 20.5, then specializes again with shape == triangle so that
no growth-set negative is covered. On the prune split that extra condition
kills the genuine heavy item square/32/bad, so pruning deletes the shape
condition and keeps only weight >= 20.5. The noisy square/31/ok is not
memorized: it is predicted as bad, while the rule stays one condition long.

The trace reader records the conditions added during growing and the ones
removed during pruning, plus each replacement/revision decision and every
description-length stop.

Compare RIPPER with other rule learners in the
[Classifier Bench](classifier_bench.md).
