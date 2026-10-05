# Bayesian Probit Classifier

`Ai4r::Classifiers::BayesianProbit` is a binary classifier built on a probit
model with a Gaussian prior over its parameters. It is meant for small-data
situations where you need to know *how much* the model knows, not just which
class it prefers.

A 0.9 probability can come from abundant evidence or from a handful of
samples that happen to be separable. The classifier keeps a Laplace
(Gaussian) approximation of the posterior after training — the posterior
mode and the covariance approximation — so both situations can be told
apart.

## Model

    P(y = 1 | x) = Phi(w . x + b)

`Phi` is the standard normal CDF, `w` the feature weights and `b` the bias
(intercept). The prior is Gaussian and fully configurable:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `prior_mean` | `0.0` | Prior mean of the feature weights |
| `prior_variance` | `10.0` | Prior variance of the feature weights |
| `bias_prior_mean` | `0.0` | Prior mean of the bias term |
| `bias_prior_variance` | `10.0` | Prior variance of the bias term |
| `max_iterations` | `100` | Newton iteration budget |
| `tolerance` | `1e-8` | Convergence tolerance (gradient infinity norm) |

The bias has its **own explicit prior**, independent of the feature weights:
it is appended as the last parameter and regularized with
`bias_prior_mean` / `bias_prior_variance`. Both variances must be positive
and finite — a proper prior is what keeps the posterior well defined under
complete separation or exactly collinear features, so infinite or
non-positive variances raise `ArgumentError`.

## Training and convergence status

Training runs Newton's method on the posterior log-density with a
backtracking line search, then inverts the posterior precision matrix at the
mode to obtain the covariance approximation. The outcome is reported
explicitly instead of silently returning `NaN`:

    classifier = Ai4r::Classifiers::BayesianProbit.new
    classifier.build(data_set)
    classifier.converged?       # => true/false
    classifier.iterations       # => 13
    classifier.failure_reason   # => nil, or e.g. "did not converge within 100 iterations"

If training failed, prediction methods raise with the failure reason.

## Predicting with parameter uncertainty

Predictions integrate over the Gaussian posterior instead of plugging the
mode back into the probit. For a posterior `N(m, S)` the predictive
probability is the closed form

    P(y = 1 | x) = Phi(a / sqrt(1 + v)),  a = m . x,  v = x' S x

    classifier.predict_proba([1.1])     # => uncertainty-aware P(y = 1)
    classifier.eval([1.1])              # => 0 or 1 (threshold 0.5)
    classifier.latent_response([1.1])   # => [a, v] latent mean and variance
    classifier.posterior_mode           # => [w..., b]
    classifier.posterior_covariance     # => (d+1) x (d+1) matrix

With scarce data `v` is large and `predict_proba` is pulled towards 0.5
compared to the mode plug-in `Phi(a)`; with abundant data the two coincide.

## Robustness

- **Complete separation**: the proper prior keeps the mode finite; training
  converges and probabilities stay in `(0, 1)`.
- **Collinear features**: the prior precision keeps the posterior precision
  positive definite, so the covariance approximation remains computable.
- **Extreme inputs**: the normal CDF and the Mills ratio `phi/Phi` are
  evaluated with tail-stable formulas (`erfc` and an asymptotic expansion),
  and the predictive ratio is formed with input scaling, so values like
  `1e8` yield finite probabilities instead of `NaN`.

## Example

See `examples/classifiers/bayesian_probit_example.rb` for a few-shot alert
triage demo, including how the latent variance shrinks as the same batch of
observations is repeatedly added.
