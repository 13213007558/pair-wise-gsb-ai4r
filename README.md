# AI4R — Artificial Intelligence for Ruby

🎓 **Welcome to AI4R**

Current version: 2.0

Your Lightweight Lab for AI and Machine Learning in Ruby.

AI4R isn’t just another machine learning library. It’s a learning playground. No black boxes and no bulky dependencies—just clean, readable Ruby implementations of core AI algorithms so you can explore, modify and really understand how they work.

## Installation

AI4R is distributed as a gem and requires Ruby 3.2 or later.

Install the gem using RubyGems:

```bash
gem install ai4r
```

Add the library to your code:

```ruby
require 'ai4r'
```

## Where to start?
- [Beginner Track](docs/learning_path_1_beginner.md) – Build core intuition for AI—step by step, in Ruby.
- [Intermediate Track](docs/learning_path_2_intermediate.md) – From "I can run a model" to "I can tune, extend, and build smart stuff that actually works."
- [Advanced Track](docs/learning_path_3_advanced.md) – Time to stop following recipes and start writing your own.


## 🧭 What’s Inside?

A quick map to AI4R’s built‑in toolkits, grouped by type. Each folder comes with examples and benchmark runners so you can dive right in.

### 🤖 Transformers – *Play with the Building Blocks of Modern LLMs*

> *“Meet your future coworker / overlord.”*

This is not a full GPT—but it is the core logic, stripped down and readable.  
AI4R ships with a bite-sized, dependency-free Transformer implementation that supports:

- **Encoder-only** mode (like BERT)
- **Decoder-only** mode (like GPT)
- **Seq2Seq** mode (like T5)

📂 Code: `lib/ai4r/neural_network/transformer.rb`
Docs: [Transformer guide](docs/transformer.md)

💡 **Try this**:  
Load up the transformer and walk through a simple forward pass.  
Everything from attention weights to layer normalization is short enough to read and understand in one go.

### 🧠 Classifiers – Make Predictions
"What’s the most likely outcome?"

You’ll find in [lib/ai4r/classifiers/](lib/ai4r/classifiers/):

- `ZeroR`, `OneR` – the simplest baselines
- `LogisticRegression`, `SimpleLinearRegression`
- `SupportVectorMachine`
- `RandomForest`, `GradientBoosting`, `SAMMEBoost`
- `MultilayerPerceptron`

Docs: [logistic_regression.md](docs/logistic_regression.md), [random_forest.md](docs/random_forest.md)

Try this: run `compare_all.rb` to benchmark classifiers on real datasets.

### 🔍 Clusterers – Find Hidden Patterns
"What belongs together?"

Includes in [lib/ai4r/clusterers/](lib/ai4r/clusterers/):

- `KMeans`
- `DBSCAN`
- `Hierarchical` clustering variants

Docs: [kmeans.md](docs/kmeans.md), [dbscan.md](docs/dbscan.md)

Try this: cluster the Iris dataset using both KMeans and DBSCAN.

### 🧬 Neural Networks – Learn From Data
"What if we build a brain?"

- Backpropagation – classic feedforward network
- Hopfield – associative memory model
- Transformer – a tiny GPT‑style block (encoder, decoder, seq2seq)

Code: `lib/ai4r/neural_network/`

Try this: open `transformer.rb` and trace each step—it’s short enough to grok in one sitting.

### 🔎 Search Algorithms – Explore Possibility Spaces
"What’s the best path?"

You’ll find in [lib/ai4r/search/](lib/ai4r/search/):

- `BreadthFirst`, `DepthFirst`, `IterativeDeepening`
- `A*`
- `MonteCarloTreeSearch`

Docs: [search_algorithms.md](docs/search_algorithms.md)

Try this: run A* and DFS on a maze and time the difference.

### 🧪 Genetic Algorithms – Evolve a Solution
"Let’s mutate our way to a better answer."

- Generic GA framework
- A Traveling Salesman Problem (TSP) chromosome

Code: `lib/ai4r/genetic_algorithm/`

Try this: tweak the mutation rate in the TSP example.

### 🧭 Reinforcement Learning – Learn by Doing
"Reward me, and I’ll improve."

- Q‑Learning
- Policy Iteration

Code: `lib/ai4r/reinforcement/`

Docs: `docs/reinforcement_learning.md`

Try this: run a grid‑world training loop and watch the agent build its own policy.

### 🕵️ Hidden Markov Models – Guess What’s Hidden
"You can’t see the states—but you can infer them."

Code: `lib/ai4r/hmm/hidden_markov_model.rb`

Docs: `docs/hmm.md`

Try this: model a weather prediction problem with hidden states and visible activities.

### 🧠 Self‑Organizing Maps – Compress Dimensions
"Can we project complex data onto a simpler map?"

- Kohonen‑style SOM

Code: `lib/ai4r/som/`

Try this: reduce high‑dimensional vectors into a 2D neuron grid and color it based on class.

## 🧪 Benchmarks: Experiment & Compare

Each algorithm family has a benchmark runner:

- `bench/classifier/`
- `bench/clusterer/`
- `bench/search/`

Shared tools in `bench/common/` make it easy to run head‑to‑head comparisons, track runtime, accuracy and more, and output clean reports.

Docs: `docs/benches_overview.md`

Try this: run `bench/search/astar_vs_dfs.rb` and explain why A* usually wins.


```bash
git clone https://github.com/SergioFierens/ai4r
cd ai4r
bundle install
ruby bench/classifier/compare_all.rb
```



## 💬 Feedback?

This library is maintained for the joy of it (and perhaps a misplaced sense of duty to Ruby). You can do whatever you want with it—it’s unlicensed. If you build something cool or just find it useful, drop a note in the [project's comments](https://github.com/SergioFierens/ai4r/discussions).
