# Bayesian Probit

`Ai4r::Classifiers::BayesianProbit` fits a binary probit classifier with a Gaussian prior over the feature coefficients and a separate, explicit Gaussian prior for the bias. The bias is always appended as the final design-matrix column and has the same prior treatment as every parameter.

Training uses Newton iterations with a backtracking line search to find the posterior mode. A Laplace approximation stores the inverse negative Hessian as `posterior_covariance`. Prediction integrates the probit link over that approximate posterior in one dimension, so parameter uncertainty changes the probability; evaluating the probit link at only the mode is not sufficient.

## Parameters

* `slope_prior_mean` – Numeric prior mean for all slopes, or an Array with one mean per feature.
* `slope_prior_variance` – Positive variance, Array of variances, or symmetric positive-definite covariance matrix.
* `bias_prior_mean` – Prior mean for the appended bias. Default `0.0`.
* `bias_prior_variance` – Positive prior variance for the appended bias. Default `100.0`.
* `max_iterations` – Maximum Newton iterations. Default `50`.
* `tolerance` – Infinity-norm convergence tolerance. Default `1e-8`.
* `integration_tolerance`, `integration_stdev`, `integration_depth` – Controls for the one-dimensional adaptive Simpson predictive integral.

## Results

* `converged`, `iterations`, and `optimization_status` describe optimization.
* `posterior_mode` contains slopes followed by bias.
* `posterior_covariance` is the Laplace covariance approximation.
* `latent_linear_response(x)` returns `[mean, variance]` for the latent response `x'b`.
* `positive_class_probability(x)` and `get_probability_map(x)` include posterior uncertainty.
* `warnings` reports complete/quasi-complete separation and rank-deficient or collinear designs. These cases remain finite because of the prior, but the result is prior-dependent.

## Example

```ruby
require 'ai4r/classifiers/bayesian_probit'
require 'ai4r/data/data_set'

data = Ai4r::Data::DataSet.new(data_items: [[-3, 0], [-2, 0], [1, 1], [2, 1]])
model = Ai4r::Classifiers::BayesianProbit.new
model.set_parameters(slope_prior_variance: 1.0, bias_prior_variance: 100.0)
model.build(data)

mean, variance = model.latent_linear_response([1])
probability = model.positive_class_probability([1])
```

The runnable script `examples/classifiers/bayesian_probit_example.rb` uses a tiny separated one-dimensional dataset, compares the integrated prediction with a mode-only probit value, and repeats the observations to show uncertainty shrinking.
