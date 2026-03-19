# reeval — Reliable and Efficient EVALuations

[![PyPI version](https://img.shields.io/pypi/v/reeval.svg)](https://pypi.org/project/reeval/)
[![Python](https://img.shields.io/pypi/pyversions/reeval.svg)](https://pypi.org/project/reeval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Theomat/reeval/blob/main/LICENSE)

**reeval** is a Python library for computing statistically-grounded sample sizes and confidence guarantees for empirical evaluations and benchmarks. It treats an evaluation as a random sample drawn from a population and provides principled, formal guarantees via the Central Limit Theorem, Bonferroni correction, and Cochran's finite-population formula.
It follows and implement every guideline mentioned in [A Hitchhiker's guide to statistical tests for assessing randomized algorithms in software engineering](https://doi.org/10.1002/stvr.1486).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Core Concepts](#core-concepts)
5. [Examples](#examples)
   - [Boolean measure — proportion / accuracy](#1-boolean-measure--proportion--accuracy)
   - [Mean measure with known standard deviation](#2-mean-measure-with-known-standard-deviation)
   - [Mean measure with unknown standard deviation (Student-t)](#3-mean-measure-with-unknown-standard-deviation-student-t)
   - [Rank measure](#4-rank-measure)
   - [Variance measure](#5-variance-measure)
   - [Computing confidence from a fixed sample size](#6-computing-confidence-from-a-fixed-sample-size)
   - [Computing absolute error from sample size and confidence](#7-computing-absolute-error-from-sample-size-and-confidence)
   - [Aggregating multiple measures with Evaluation](#8-aggregating-multiple-measures-with-evaluation)
   - [Type II error — power analysis](#9-type-ii-error--power-analysis)
   - [Finite population correction](#10-finite-population-correction)
   - [Filtered populations](#11-filtered-populations)
   - [Global sample size solver and reporting](#12-global-sample-size-solver-and-reporting)
   - [Categorical measures](#13-categorical-measures)
   - [Hypothesis test — boolean data (Fisher's exact)](#14-hypothesis-test--boolean-data-fishers-exact)
   - [Hypothesis test — continuous data (Welch's t-test)](#15-hypothesis-test--continuous-data-welchs-t-test)
   - [Hypothesis test — paired data (Wilcoxon signed-rank)](#16-hypothesis-test--paired-data-wilcoxon-signed-rank)
   - [Hypothesis test — ranked data (Mann-Whitney U)](#17-hypothesis-test--ranked-data-mann-whitney-u)
   - [Effect sizes — Vargha-Delaney A12](#18-effect-sizes--vargha-delaney-a12)
   - [Effect sizes — odds ratio with confidence interval](#19-effect-sizes--odds-ratio-with-confidence-interval)
6. [Citing](#citing)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

Designing a reliable evaluation requires answering a deceptively hard question: *how many instances do I need?* A sample that is too small leads to conclusions that may not generalise; a sample that is unnecessarily large wastes resources.

**reeval** formalises this question. Every evaluation is a statistical estimation problem: a set of measures are computed over a random sample drawn from some population, and the goal is to bound the estimation error with a given confidence. Given a desired confidence level and an acceptable absolute (or relative) error, **reeval** computes the minimum sample size required. Conversely, given a fixed sample size, it computes the confidence or error bound that is actually achieved.

The library supports four measure types (boolean, mean, rank, variance), multiple simultaneous measures with automatic Bonferroni correction, finite populations via Cochran's formula, and hierarchical evaluations where one evaluation filters the population for the next.

---

## Features

- **Sample size computation** — Given a confidence level and an error tolerance, compute the minimum sample size required for any supported measure type.
- **Confidence computation** — Given a fixed sample size, compute the confidence level (or statistical power) achieved for each measure.
- **Absolute / relative error computation** — Given a sample size and a confidence, compute the error bound guaranteed for each measure.
- **Four measure types**:
  - `BooleanMeasure` — for binary outcomes such as accuracy, pass/fail rates, or proportions; uses a normal approximation with configurable or worst-case (0.5) standard deviation.
  - `MeanMeasure` — for continuous values such as scores, running times, or costs; supports both known variance (normal distribution) and unknown variance (iterative Student-t distribution).
  - `RankMeasure` — for ordinal rankings; automatically derives the standard deviation from the number of ranks using the discrete uniform distribution.
  - `VarianceMeasure` — for estimating variance itself, using a relative error bound.
- **Categorical measures** — a factory that expands a single categorical variable into one `BooleanMeasure` per category.
- **Type I and Type II error control** — switch between controlling the false-positive rate (Type I) and the false-negative rate / statistical power (Type II).
- **Bonferroni correction** — automatically applied across measures and evaluation repeats to control the family-wise error rate.
- **Finite population correction** — Cochran's formula reduces the required sample size when the population is bounded.
- **Filtered populations** — model hierarchical evaluations where a second evaluation runs on a subset identified by a first evaluation; the library propagates confidence and conservatively estimates the filtered population size.
- **Global sample size solver** — iteratively resolves sample size requirements for chains of dependent evaluations.
- **Global reporting helpers** — evaluate achieved confidence/power and absolute errors from externally supplied sample-size assignments.
- **Hypothesis tests with effect sizes** — per-measure two-sample tests returning p-value, effect size, and a confidence interval:
  - Boolean: Fisher's exact test, odds ratio with Woolf logit CI.
  - Mean / Rank: Welch's t-test or Mann-Whitney U, Vargha-Delaney A12 with normal-approximation CI.
  - Paired mean data: Wilcoxon signed-rank test.

---

## Installation

```bash
pip install reeval
```

**Requirements**: Python >= 3.11, SciPy >= 1.17.

---

## Core Concepts

| Concept | Description |
| --- | --- |
| **Measure** | A quantity computed from evaluation instances (e.g. accuracy, mean score). |
| **Absolute error** `δ` | The maximum acceptable estimation error for the measure, this type of error is additive. |
| **Relative error** `δ_rel` | Like absolute error but multiplicative error; used for `VarianceMeasure`. |
| **Error rate** `α` or `β` | The allowed probability of exceeding the error bound (Type I) or missing a true effect (Type II). |
| **Confidence** `1 − α` | The probability that the estimate is within the error bound. |
| **Power** `1 − β` | The probability of detecting a true effect of the specified magnitude. |
| **Multiple hypothesis correction** | Adjusts the error budget by the number of simultaneous comparisons to control the family-wise rate of either confidence or power. |
| **FInite population correction** | Adjusts the sample size downward for finite populations. |
| **Filtered population** | A subset of the population defined by a previous evaluation, used for hierarchical designs. |
| **Stage success probability** | Success probability already consumed by an upstream stage such as filter estimation; downstream guarantees are composed with it. |

---

## Examples

### 1. Boolean measure — proportion / accuracy

Compute how many instances are needed to estimate a binary proportion (e.g. the accuracy of a model) within ±0.02 at 95% confidence.

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

n = measure.compute_sample_size(ErrorControl.type_i(0.05))
print(f"Required sample size: {n}")
# => Required sample size: 2401
```

The default standard deviation is 0.5 (worst case for a Bernoulli variable), giving a conservative estimate. If you have prior knowledge that the proportion lies near 0.9, supply the corresponding std:

```python
import math

# std for a Bernoulli(p) is sqrt(p*(1-p)); here p ~ 0.9
measure = BooleanMeasure(name="accuracy", std=math.sqrt(0.9 * 0.1), absolute_error=0.02)
n = measure.compute_sample_size(ErrorControl.type_i(0.05))
print(f"Required sample size: {n}")
# => Required sample size: 865  (smaller because the variance is lower)
```

You can also account for imperfect labels by setting `sensitivity` and `specificity`. The library uses the standard prevalence correction
`p = (q + specificity - 1) / (sensitivity + specificity - 1)`, so the effective standard deviation is inflated by the identifiability factor `1 / (sensitivity + specificity - 1)`:

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

# Labels are not perfect:
# - 97% sensitivity
# - 98% specificity
measure = BooleanMeasure(
    name="accuracy",
    absolute_error=0.02,
    sensitivity=0.97,
    specificity=0.98,
)

n = measure.compute_sample_size(ErrorControl.type_i(0.05))
print(f"Required sample size with noisy labels: {n}")
```

If you want to model only one aspect of label quality, you can set the other one to `1.0`:

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

# Only account for sensitivity
sensitivity_measure = BooleanMeasure(
    name="accuracy",
    absolute_error=0.02,
    sensitivity=0.9,
)

# Only account for specificity
specificity_measure = BooleanMeasure(
    name="accuracy",
    absolute_error=0.02,
    specificity=0.95,
)

for label, measure in [
    ("sensitivity", sensitivity_measure),
    ("specificity", specificity_measure),
]:
    n = measure.compute_sample_size(ErrorControl.type_i(0.05))
    print(f"{label}: {n}")
```

Lower `sensitivity` or `specificity` increases the effective uncertainty in this model and therefore increases the computed sample size. As `sensitivity + specificity` approaches `1`, the corrected prevalence becomes non-identifiable and the required sample size diverges.

---

### 2. Mean measure with known standard deviation

Estimate the mean running time of a solver within ±0.5 seconds at 99% confidence, assuming a known standard deviation of 2 seconds.

```python
from reeval.measures import MeanMeasure
from reeval import ErrorControl

measure = MeanMeasure(name="runtime", std=2.0, absolute_error=0.5)

n = measure.compute_sample_size(ErrorControl.type_i(0.01))
print(f"Required sample size: {n}")
```

---

### 3. Mean measure with unknown standard deviation (Student-t)

When the standard deviation is not known in advance, omit `std`. The library uses an iterative Student-t formula that is self-consistent (the degrees of freedom depend on the unknown sample size).

```python
from reeval.measures import MeanMeasure
from reeval import ErrorControl

# No std provided: Student-t distribution with unknown variance
measure = MeanMeasure(name="f1_score", absolute_error=0.05)

n = measure.compute_sample_size(ErrorControl.type_i(0.05))
print(f"Required sample size (Student-t): {n}")
```

---

### 4. Rank measure

Estimate the mean rank assigned to items on a 1–5 Likert scale within ±0.3 rank points at 95% confidence. The standard deviation is derived automatically from the number of ranks using the discrete uniform distribution: `σ = sqrt((k²−1)/12)`.

```python
from reeval.measures import RankMeasure
from reeval import ErrorControl

measure = RankMeasure(name="user_rating", max_rank=5, absolute_error=0.3)

n = measure.compute_sample_size(ErrorControl.type_i(0.05))
print(f"Required sample size: {n}")
```

---

### 5. Variance measure

Estimate the variance of a quantity within a ±10% relative error at 95% confidence. Because the target depends on the unknown true variance, `VarianceMeasure` works with a relative error rather than an absolute one.

```python
from reeval.measures import VarianceMeasure
from reeval import ErrorControl

measure = VarianceMeasure(name="score_variance", relative_error=0.10)

n = measure.compute_sample_size(ErrorControl.type_i(0.05))
print(f"Required sample size: {n}")
```

To retrieve the relative error bound achieved by a fixed sample size:

```python
rel_err = measure.compute_relative_error(sample_size=400, error_control=ErrorControl.type_i(0.05))
print(f"Relative error at n=400: +/-{rel_err:.3f}")
```

---

### 6. Computing confidence from a fixed sample size

If the sample size is already fixed (e.g. by resource constraints), compute what confidence is actually achieved.

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

confidence = measure.compute_error_probability(sample_size=1000, error_control=ErrorControl.type_i(0.05))
print(f"Achieved confidence at n=1000: {confidence:.3f}")
# => Achieved confidence at n=1000: 0.898
```

The same method with `ErrorControl.type_ii(...)` returns achieved statistical power for detecting the configured effect size at significance level `α`:

```python
power = measure.compute_error_probability(
    sample_size=1000,
    error_control=ErrorControl.type_ii(0.20, significance_level=0.05),
)
print(f"Achieved power at n=1000: {power:.3f}")
```

---

### 7. Computing absolute error from sample size and confidence

Given a fixed sample size and a desired confidence level, compute the error bound that is guaranteed.

```python
from reeval.measures import MeanMeasure
from reeval import ErrorControl

measure = MeanMeasure(name="score", std=1.5, absolute_error=0.1)

abs_error = measure.compute_absolute_error(
    sample_size=500, error_control=ErrorControl.type_i(0.05)
)
print(f"Guaranteed absolute error at n=500: +/-{abs_error:.4f}")
```

---

### 8. Aggregating multiple measures with Evaluation

`Evaluation` aggregates several measures defined on the same sample. Bonferroni correction is applied automatically so that the family-wise confidence covers all measures simultaneously.

```python
from reeval import ErrorControl, Evaluation
from reeval.measures import BooleanMeasure, MeanMeasure
from reeval.population import InfinitePopulation

accuracy = BooleanMeasure(name="accuracy", absolute_error=0.02)
latency  = MeanMeasure(name="latency_ms", std=50.0, absolute_error=5.0)

eval = Evaluation(
    measures=[accuracy, latency],
    population=InfinitePopulation(),
    error_control=ErrorControl.type_i(0.05),
)

n = eval.compute_sample_size()
print(f"Required sample size to satisfy both measures: {n}")
```

The required sample size is the maximum across all measures after Bonferroni correction, so the most demanding measure drives the result.

---

### 9. Type II error — power analysis

Switch to `ErrorControl.type_ii(...)` to do classical power analysis instead of two-sided confidence calculation.

Academically, power is a property of a hypothesis test under an alternative. It depends on both:

- the rejection threshold `α`
- the alternative effect size to detect

For the normal-approximation measures in `reeval`, TYPE_II now uses the standard two-sided z-test approximation

`n ≈ ((z_{1-α/2} + z_{1-β}) σ / δ)^2`

where `δ` is the configured absolute or relative effect size. The corresponding minimum detectable effect is

`δ ≈ (z_{1-α/2} + z_{1-β}) σ / √n`.

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

n_type_i  = measure.compute_sample_size(ErrorControl.type_i(0.05))
n_type_ii = measure.compute_sample_size(
    ErrorControl.type_ii(0.05, significance_level=0.05)
)

print(f"n (Type I,  α=0.05): {n_type_i}")
print(f"n (Type II, α=0.05, β=0.05): {n_type_ii}")
```

At equal nominal error rates, TYPE_II usually requires more samples than TYPE_I because power must satisfy both the significance threshold and the miss-rate target.

---

### 10. Finite population correction

When sampling from a bounded population, Cochran's formula reduces the required sample size. This is relevant when the population contains, say, 5 000 instances and you would otherwise need 2 400.

```python
from reeval import Evaluation
from reeval.measures import BooleanMeasure
from reeval import ErrorControl
from reeval.population import InfinitePopulation, FinitePopulation

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

eval_infinite = Evaluation(
    measures=[measure],
    population=InfinitePopulation(),
    error_control=ErrorControl.type_i(0.05),
)

eval_finite = Evaluation(
    measures=[measure],
    population=FinitePopulation(size=5000),
    error_control=ErrorControl.type_i(0.05),
)

print(f"n (infinite population):    {eval_infinite.compute_sample_size()}")
print(f"n (finite population N=5000): {eval_finite.compute_sample_size()}")
# The finite version is smaller due to Cochran's correction.
```

---

### 11. Filtered populations

`FilteredPopulation` models the common academic workflow where one evaluation first estimates the prevalence of a boolean property, and a second evaluation is then run only on the retained subpopulation.

For a finite source population of size `N`, empirical prevalence `p_hat`, and filter absolute error `eps`, the library uses the conservative upper bound

`|P_filtered| <= ceil(N * min(1, p_hat + eps))`.

At the same time, the filter estimation step already consumes part of the statistical guarantee. The downstream evaluation therefore tightens its own error budget automatically so that the final joint guarantee remains valid.

```python
from reeval import ErrorControl, Evaluation
from reeval.measures import BooleanMeasure, MeanMeasure
from reeval.population import FilteredPopulation, FinitePopulation

is_bug = BooleanMeasure(name="is_bug", absolute_error=0.05)
severity = MeanMeasure(name="severity", std=1.2, absolute_error=0.2)

source_population = FinitePopulation(size=10_000)

bug_population = FilteredPopulation(
    source_population=source_population,
    error_control=ErrorControl.type_i(0.01),  # guarantee from the filter-estimation stage
    filter_measure=is_bug,
    empirical_proportion=0.30,                # observed bug prevalence
)

bug_eval = Evaluation(
    measures=[severity],
    population=bug_population,
    error_control=ErrorControl.type_i(0.05),
)

print(f"Conservative filtered population size: {bug_population.get_size()}")
print(f"Required sample size inside filtered population: {bug_eval.compute_sample_size()}")
```

Use `Population.filter(...)` if you prefer a shorter construction:

```python
bug_population = source_population.filter(
    measure=is_bug,
    empirical_proportion=0.30,
    error_control=ErrorControl.type_i(0.01),
)
```

---

### 12. Global sample size solver and reporting

For chained evaluations, local sample sizes are not enough. If a downstream evaluation needs `n` retained items and the filter prevalence is only known up to a conservative lower bound `p_lower`, the upstream stage must inspect at least `ceil(n / p_lower)` source items.

`compute_global_sample_sizes(...)` resolves these dependencies to a fixed point.

```python
from reeval import (
    ErrorControl,
    Evaluation,
    compute_global_absolute_errors,
    compute_global_error_probabilities,
    compute_global_sample_sizes,
)
from reeval.measures import BooleanMeasure, MeanMeasure
from reeval.population import FilteredPopulation, FinitePopulation

screening = Evaluation(
    measures=[BooleanMeasure(name="is_bug", absolute_error=0.05)],
    population=FinitePopulation(size=10_000),
    error_control=ErrorControl.type_i(0.20),
)

bug_population = FilteredPopulation(
    source_population=screening.population,
    error_control=ErrorControl.type_i(0.01),
    filter_measure=BooleanMeasure(name="is_bug", absolute_error=0.05),
    empirical_proportion=0.40,
)

severity = Evaluation(
    measures=[MeanMeasure(name="severity", std=1.0, absolute_error=0.02)],
    population=bug_population,
    error_control=ErrorControl.type_i(0.05),
)

sample_sizes = compute_global_sample_sizes([screening, severity])
error_probabilities = compute_global_error_probabilities(
    [screening, severity],
    sample_sizes=sample_sizes,
)
absolute_errors = compute_global_absolute_errors(
    [screening, severity],
    sample_sizes=sample_sizes,
)

print(sample_sizes[screening])
print(error_probabilities[severity][0])  # total achieved confidence/power
print(absolute_errors[severity]["severity"])
```

The reporting helpers intentionally take `sample_sizes` as an explicit argument. This separates design from reporting: you may use the library's global solver, or supply externally chosen sample sizes.

---

### 13. Categorical measures

`CategoricalMeasures` is a factory that creates one `BooleanMeasure` per category for a categorical variable, all sharing the same error parameters.

```python
from reeval.measures import CategoricalMeasures
from reeval import ErrorControl

# Estimate the proportion in each of 4 sentiment classes.
sentiment_measures = CategoricalMeasures(
    name="sentiment",
    categories=4,
    absolute_error=0.03,
)

for m in sentiment_measures:
    n = m.compute_sample_size(ErrorControl.type_i(0.05))
    print(f"  {m.name}: n = {n}")
# sentiment_0: n = ...
# sentiment_1: n = ...
# ...
```

When these measures are combined in an `Evaluation`, Bonferroni correction accounts for all four simultaneous proportion estimates automatically.

---

### 14. Hypothesis test — boolean data (Fisher's exact)

`BooleanMeasure.test_different` runs Fisher's exact test and returns the p-value, the odds ratio as effect size, and a confidence interval using the Woolf logit method.

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

measure = BooleanMeasure(name="pass_rate", absolute_error=0.05)

# System A passes 80 out of 100; system B passes 60 out of 100.
sample_a = [True] * 80 + [False] * 20
sample_b = [True] * 60 + [False] * 40

p_value, odds_ratio, ci = measure.test_different(
    sample_a, sample_b, error_control=ErrorControl.type_i(0.05)
)

print(f"p-value:    {p_value:.4f}")
print(f"Odds ratio: {odds_ratio:.3f}")
print(f"95% CI:     ({ci[0]:.3f}, {ci[1]:.3f})")
```

Use `ErrorControl.type_ii(0.05, significance_level=0.05)` to obtain the alternative CI style currently exposed by the test helpers:

```python
_, _, ci_power = measure.test_different(
    sample_a,
    sample_b,
    error_control=ErrorControl.type_ii(0.05, significance_level=0.05),
)
print(f"Power CI: ({ci_power[0]:.3f}, {ci_power[1]:.3f})")
```

---

### 15. Hypothesis test — continuous data (Welch's t-test)

`MeanMeasure.test_different` uses Welch's t-test (which does not assume equal variances) and reports Vargha and Delaney's A12 as the effect size.

```python
import random
from reeval.measures import MeanMeasure
from reeval import ErrorControl

random.seed(42)
measure = MeanMeasure(name="score", std=1.0, absolute_error=0.1)

sample_a = [random.gauss(5.0, 1.0) for _ in range(200)]
sample_b = [random.gauss(5.5, 1.0) for _ in range(200)]

p_value, a12, ci = measure.test_different(
    sample_a, sample_b, error_control=ErrorControl.type_i(0.05)
)

print(f"p-value: {p_value:.4f}")
print(f"A12:     {a12:.3f}  (0.5 = no difference, 1.0 = A always > B)")
print(f"95% CI:  ({ci[0]:.3f}, {ci[1]:.3f})")
```

A12 = P(X > Y) + 0.5 * P(X = Y). A value of 0.5 means no stochastic ordering; values near 0 or 1 indicate a strong directional effect.

---

### 16. Hypothesis test — paired data (Wilcoxon signed-rank)

When both samples are measured on the same instances (e.g. two systems evaluated on the same benchmark items), use the paired variant based on the Wilcoxon signed-rank test.

```python
import random
from reeval.measures import MeanMeasure
from reeval import ErrorControl

random.seed(0)
measure = MeanMeasure(name="score", std=1.0, absolute_error=0.1)

# Same 150 items evaluated by two systems; scores are positively correlated.
base     = [random.gauss(5.0, 1.0) for _ in range(150)]
sample_a = [x + random.gauss(0.0, 0.2) for x in base]
sample_b = [x + random.gauss(0.3, 0.2) for x in base]

p_value, a12, ci = measure.test_different_paired_data(
    sample_a, sample_b, error_control=ErrorControl.type_i(0.05)
)

print(f"Wilcoxon p-value: {p_value:.4f}")
print(f"A12:              {a12:.3f}")
print(f"95% CI:           ({ci[0]:.3f}, {ci[1]:.3f})")
```

---

### 17. Hypothesis test — ranked data (Mann-Whitney U)

`RankMeasure.test_different` uses the Mann-Whitney U test for comparing rank distributions and returns A12 as the effect size.

```python
import random
from reeval.measures import RankMeasure
from reeval import ErrorControl

random.seed(7)
measure = RankMeasure(name="preference", max_rank=5, absolute_error=0.5)

# Users rated system A and system B on a 1–5 scale.
ratings_a = [random.randint(3, 5) for _ in range(100)]
ratings_b = [random.randint(1, 4) for _ in range(100)]

p_value, a12, ci = measure.test_different(
    ratings_a, ratings_b, error_control=ErrorControl.type_i(0.05)
)

print(f"p-value: {p_value:.4f}")
print(f"A12:     {a12:.3f}")
print(f"95% CI:  ({ci[0]:.3f}, {ci[1]:.3f})")
```

---

### 18. Effect sizes — Vargha-Delaney A12

A12 is a non-parametric, robust effect size for continuous and ordinal data. It estimates the probability that a randomly drawn observation from one group exceeds one from the other. All mean and rank tests return it automatically (see examples 15–17).

Key interpretation thresholds:

| A12 | Interpretation |
| --- | --- |
| 0.50 | No difference |
| 0.56 | Small effect |
| 0.64 | Medium effect |
| 0.71 | Large effect |

A12 and 1 − A12 are complementary: swapping the two samples turns A12 into 1 − A12.

---

### 19. Effect sizes — odds ratio with confidence interval

For boolean measures, the odds ratio measures how much more (or less) likely outcome = True is in one group versus the other. The Woolf logit method is used for the confidence interval; degenerate tables (any cell = 0) produce the interval (0, inf).

```python
from reeval.measures import BooleanMeasure
from reeval import ErrorControl

measure = BooleanMeasure(name="detection", absolute_error=0.05)

# 45/50 detected in group A, 30/50 detected in group B.
sample_a = [True] * 45 + [False] * 5
sample_b = [True] * 30 + [False] * 20

p_value, or_val, ci = measure.test_different(
    sample_a, sample_b, error_control=ErrorControl.type_i(0.05)
)

print(f"p-value:    {p_value:.4f}")
print(f"Odds ratio: {or_val:.3f}  (1.0 = no difference)")
print(f"95% CI:     ({ci[0]:.3f}, {ci[1]:.3f})")
```

---

## Citing

If you use **reeval** in your research, please cite:

```bibtex
@software{reeval2026,
  author    = {Matricon, Théo},
  title     = {{reeval}: Reliable and Efficient EVALuations},
  year      = {2026},
  url       = {https://github.com/Theomat/reeval},
  version   = {0.1.0},
  note      = {Python package for statistically-grounded sample size computation and evaluation guarantees},
}
```

---

## Contributing

Bug reports and feature requests are welcome — please open an issue on the [GitHub issue tracker](https://github.com/Theomat/reeval/issues).

If you'd like to contribute code, open a pull request against `main`. Please make sure existing tests pass (`pytest tests/`) and add tests for any new behaviour.

---

## License

This project is licensed under the [MIT License](LICENSE).
