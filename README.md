# reeval — Reliable and Efficient EVALuations

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
   - [Categorical measures](#11-categorical-measures)
   - [Hypothesis test — boolean data (Fisher's exact)](#12-hypothesis-test--boolean-data-fishers-exact)
   - [Hypothesis test — continuous data (Welch's t-test)](#13-hypothesis-test--continuous-data-welchs-t-test)
   - [Hypothesis test — paired data (Wilcoxon signed-rank)](#14-hypothesis-test--paired-data-wilcoxon-signed-rank)
   - [Hypothesis test — ranked data (Mann-Whitney U)](#15-hypothesis-test--ranked-data-mann-whitney-u)
   - [Effect sizes — Vargha-Delaney A12](#16-effect-sizes--vargha-delaney-a12)
   - [Effect sizes — odds ratio with confidence interval](#17-effect-sizes--odds-ratio-with-confidence-interval)
6. [Citing](#citing)

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
- **Hypothesis tests with effect sizes** — per-measure two-sample tests returning p-value, effect size, and a confidence interval:
  - Boolean: Fisher's exact test, odds ratio with Woolf logit CI.
  - Mean / Rank: Welch's t-test or Mann-Whitney U, Vargha-Delaney A12 with normal-approximation CI.
  - Paired mean data: Wilcoxon signed-rank test.

---

## Installation

```bash
pip install git+https://github.com/Theomat/reeval.git
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

---

## Examples

### 1. Boolean measure — proportion / accuracy

Compute how many instances are needed to estimate a binary proportion (e.g. the accuracy of a model) within ±0.02 at 95% confidence.

```python
from reeval.measures import BooleanMeasure
from reeval.error_type import ErrorType

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

n = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
print(f"Required sample size: {n}")
# => Required sample size: 2401
```

The default standard deviation is 0.5 (worst case for a Bernoulli variable), giving a conservative estimate. If you have prior knowledge that the proportion lies near 0.9, supply the corresponding std:

```python
import math

# std for a Bernoulli(p) is sqrt(p*(1-p)); here p ~ 0.9
measure = BooleanMeasure(name="accuracy", std=math.sqrt(0.9 * 0.1), absolute_error=0.02)
n = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
print(f"Required sample size: {n}")
# => Required sample size: 865  (smaller because the variance is lower)
```

---

### 2. Mean measure with known standard deviation

Estimate the mean running time of a solver within ±0.5 seconds at 99% confidence, assuming a known standard deviation of 2 seconds.

```python
from reeval.measures import MeanMeasure
from reeval.error_type import ErrorType

measure = MeanMeasure(name="runtime", std=2.0, absolute_error=0.5)

n = measure.compute_sample_size(error=0.01, error_type=ErrorType.TYPE_I)
print(f"Required sample size: {n}")
```

---

### 3. Mean measure with unknown standard deviation (Student-t)

When the standard deviation is not known in advance, omit `std`. The library uses an iterative Student-t formula that is self-consistent (the degrees of freedom depend on the unknown sample size).

```python
from reeval.measures import MeanMeasure
from reeval.error_type import ErrorType

# No std provided: Student-t distribution with unknown variance
measure = MeanMeasure(name="f1_score", absolute_error=0.05)

n = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
print(f"Required sample size (Student-t): {n}")
```

---

### 4. Rank measure

Estimate the mean rank assigned to items on a 1–5 Likert scale within ±0.3 rank points at 95% confidence. The standard deviation is derived automatically from the number of ranks using the discrete uniform distribution: `σ = sqrt((k²−1)/12)`.

```python
from reeval.measures import RankMeasure
from reeval.error_type import ErrorType

measure = RankMeasure(name="user_rating", max_rank=5, absolute_error=0.3)

n = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
print(f"Required sample size: {n}")
```

---

### 5. Variance measure

Estimate the variance of a quantity within a ±10% relative error at 95% confidence. Because the target depends on the unknown true variance, `VarianceMeasure` works with a relative error rather than an absolute one.

```python
from reeval.measures import VarianceMeasure
from reeval.error_type import ErrorType

measure = VarianceMeasure(name="score_variance", relative_error=0.10)

n = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
print(f"Required sample size: {n}")
```

To retrieve the relative error bound achieved by a fixed sample size:

```python
rel_err = measure.compute_relative_error(sample_size=400, error=0.05, error_type=ErrorType.TYPE_I)
print(f"Relative error at n=400: +/-{rel_err:.3f}")
```

---

### 6. Computing confidence from a fixed sample size

If the sample size is already fixed (e.g. by resource constraints), compute what confidence is actually achieved.

```python
from reeval.measures import BooleanMeasure
from reeval.error_type import ErrorType

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

confidence = measure.compute_error_probability(sample_size=1000, error_type=ErrorType.TYPE_I)
print(f"Achieved confidence at n=1000: {confidence:.3f}")
# => Achieved confidence at n=1000: 0.898
```

The same method with `ErrorType.TYPE_II` returns the achieved statistical power:

```python
power = measure.compute_error_probability(sample_size=1000, error_type=ErrorType.TYPE_II)
print(f"Achieved power at n=1000: {power:.3f}")
```

---

### 7. Computing absolute error from sample size and confidence

Given a fixed sample size and a desired confidence level, compute the error bound that is guaranteed.

```python
from reeval.measures import MeanMeasure
from reeval.error_type import ErrorType

measure = MeanMeasure(name="score", std=1.5, absolute_error=0.1)

abs_error = measure.compute_absolute_error(
    sample_size=500, error=0.05, error_type=ErrorType.TYPE_I
)
print(f"Guaranteed absolute error at n=500: +/-{abs_error:.4f}")
```

---

### 8. Aggregating multiple measures with Evaluation

`Evaluation` aggregates several measures defined on the same sample. Bonferroni correction is applied automatically so that the family-wise confidence covers all measures simultaneously.

```python
from reeval import Evaluation
from reeval.measures import BooleanMeasure, MeanMeasure
from reeval.error_type import ErrorType
from reeval.population import InfinitePopulation

accuracy = BooleanMeasure(name="accuracy", absolute_error=0.02)
latency  = MeanMeasure(name="latency_ms", std=50.0, absolute_error=5.0)

eval = Evaluation(
    measures=[accuracy, latency],
    population=InfinitePopulation(),
    error_control=(0.05, ErrorType.TYPE_I),
)

n = eval.compute_sample_size()
print(f"Required sample size to satisfy both measures: {n}")
```

The required sample size is the maximum across all measures after Bonferroni correction, so the most demanding measure drives the result.

---

### 9. Type II error — power analysis

Switch to `ErrorType.TYPE_II` to control the false-negative rate (ensure sufficient statistical power) instead of the false-positive rate. Type II control uses a one-sided quantile and therefore typically requires fewer samples.

```python
from reeval.measures import BooleanMeasure
from reeval.error_type import ErrorType

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

n_type_i  = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
n_type_ii = measure.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_II)

print(f"n (Type I,  α=0.05): {n_type_i}")
print(f"n (Type II, β=0.05): {n_type_ii}  # one-sided -> fewer samples")
```

---

### 10. Finite population correction

When sampling from a bounded population, Cochran's formula reduces the required sample size. This is relevant when the population contains, say, 5 000 instances and you would otherwise need 2 400.

```python
from reeval import Evaluation
from reeval.measures import BooleanMeasure
from reeval.error_type import ErrorType
from reeval.population import InfinitePopulation, FinitePopulation

measure = BooleanMeasure(name="accuracy", absolute_error=0.02)

eval_infinite = Evaluation(
    measures=[measure],
    population=InfinitePopulation(),
    error_control=(0.05, ErrorType.TYPE_I),
)

eval_finite = Evaluation(
    measures=[measure],
    population=FinitePopulation(size=5000),
    error_control=(0.05, ErrorType.TYPE_I),
)

print(f"n (infinite population):    {eval_infinite.compute_sample_size()}")
print(f"n (finite population N=5000): {eval_finite.compute_sample_size()}")
# The finite version is smaller due to Cochran's correction.
```

---

### 11. Categorical measures

`CategoricalMeasures` is a factory that creates one `BooleanMeasure` per category for a categorical variable, all sharing the same error parameters.

```python
from reeval.measures import CategoricalMeasures
from reeval.error_type import ErrorType

# Estimate the proportion in each of 4 sentiment classes.
sentiment_measures = CategoricalMeasures(
    name="sentiment",
    categories=4,
    absolute_error=0.03,
)

for m in sentiment_measures:
    n = m.compute_sample_size(error=0.05, error_type=ErrorType.TYPE_I)
    print(f"  {m.name}: n = {n}")
# sentiment_0: n = ...
# sentiment_1: n = ...
# ...
```

When these measures are combined in an `Evaluation`, Bonferroni correction accounts for all four simultaneous proportion estimates automatically.

---

### 12. Hypothesis test — boolean data (Fisher's exact)

`BooleanMeasure.test_different` runs Fisher's exact test and returns the p-value, the odds ratio as effect size, and a confidence interval using the Woolf logit method.

```python
from reeval.measures import BooleanMeasure
from reeval.error_type import ErrorType

measure = BooleanMeasure(name="pass_rate", absolute_error=0.05)

# System A passes 80 out of 100; system B passes 60 out of 100.
sample_a = [True] * 80 + [False] * 20
sample_b = [True] * 60 + [False] * 40

p_value, odds_ratio, ci = measure.test_different(
    sample_a, sample_b, error=0.05, error_type=ErrorType.TYPE_I
)

print(f"p-value:    {p_value:.4f}")
print(f"Odds ratio: {odds_ratio:.3f}")
print(f"95% CI:     ({ci[0]:.3f}, {ci[1]:.3f})")
```

Use `ErrorType.TYPE_II` to obtain a tighter, power-focused confidence interval (one-sided quantile):

```python
_, _, ci_power = measure.test_different(
    sample_a, sample_b, error=0.05, error_type=ErrorType.TYPE_II
)
print(f"Power CI: ({ci_power[0]:.3f}, {ci_power[1]:.3f})")
```

---

### 13. Hypothesis test — continuous data (Welch's t-test)

`MeanMeasure.test_different` uses Welch's t-test (which does not assume equal variances) and reports Vargha and Delaney's A12 as the effect size.

```python
import random
from reeval.measures import MeanMeasure
from reeval.error_type import ErrorType

random.seed(42)
measure = MeanMeasure(name="score", std=1.0, absolute_error=0.1)

sample_a = [random.gauss(5.0, 1.0) for _ in range(200)]
sample_b = [random.gauss(5.5, 1.0) for _ in range(200)]

p_value, a12, ci = measure.test_different(
    sample_a, sample_b, error=0.05, error_type=ErrorType.TYPE_I
)

print(f"p-value: {p_value:.4f}")
print(f"A12:     {a12:.3f}  (0.5 = no difference, 1.0 = A always > B)")
print(f"95% CI:  ({ci[0]:.3f}, {ci[1]:.3f})")
```

A12 = P(X > Y) + 0.5 * P(X = Y). A value of 0.5 means no stochastic ordering; values near 0 or 1 indicate a strong directional effect.

---

### 14. Hypothesis test — paired data (Wilcoxon signed-rank)

When both samples are measured on the same instances (e.g. two systems evaluated on the same benchmark items), use the paired variant based on the Wilcoxon signed-rank test.

```python
import random
from reeval.measures import MeanMeasure
from reeval.error_type import ErrorType

random.seed(0)
measure = MeanMeasure(name="score", std=1.0, absolute_error=0.1)

# Same 150 items evaluated by two systems; scores are positively correlated.
base     = [random.gauss(5.0, 1.0) for _ in range(150)]
sample_a = [x + random.gauss(0.0, 0.2) for x in base]
sample_b = [x + random.gauss(0.3, 0.2) for x in base]

p_value, a12, ci = measure.test_different_paired_data(
    sample_a, sample_b, error=0.05, error_type=ErrorType.TYPE_I
)

print(f"Wilcoxon p-value: {p_value:.4f}")
print(f"A12:              {a12:.3f}")
print(f"95% CI:           ({ci[0]:.3f}, {ci[1]:.3f})")
```

---

### 15. Hypothesis test — ranked data (Mann-Whitney U)

`RankMeasure.test_different` uses the Mann-Whitney U test for comparing rank distributions and returns A12 as the effect size.

```python
import random
from reeval.measures import RankMeasure
from reeval.error_type import ErrorType

random.seed(7)
measure = RankMeasure(name="preference", max_rank=5, absolute_error=0.5)

# Users rated system A and system B on a 1–5 scale.
ratings_a = [random.randint(3, 5) for _ in range(100)]
ratings_b = [random.randint(1, 4) for _ in range(100)]

p_value, a12, ci = measure.test_different(
    ratings_a, ratings_b, error=0.05, error_type=ErrorType.TYPE_I
)

print(f"p-value: {p_value:.4f}")
print(f"A12:     {a12:.3f}")
print(f"95% CI:  ({ci[0]:.3f}, {ci[1]:.3f})")
```

---

### 16. Effect sizes — Vargha-Delaney A12

A12 is a non-parametric, robust effect size for continuous and ordinal data. It estimates the probability that a randomly drawn observation from one group exceeds one from the other. All mean and rank tests return it automatically (see examples 13–15).

Key interpretation thresholds:

| A12 | Interpretation |
| --- | --- |
| 0.50 | No difference |
| 0.56 | Small effect |
| 0.64 | Medium effect |
| 0.71 | Large effect |

A12 and 1 − A12 are complementary: swapping the two samples turns A12 into 1 − A12.

---

### 17. Effect sizes — odds ratio with confidence interval

For boolean measures, the odds ratio measures how much more (or less) likely outcome = True is in one group versus the other. The Woolf logit method is used for the confidence interval; degenerate tables (any cell = 0) produce the interval (0, inf).

```python
from reeval.measures import BooleanMeasure
from reeval.error_type import ErrorType

measure = BooleanMeasure(name="detection", absolute_error=0.05)

# 45/50 detected in group A, 30/50 detected in group B.
sample_a = [True] * 45 + [False] * 5
sample_b = [True] * 30 + [False] * 20

p_value, or_val, ci = measure.test_different(
    sample_a, sample_b, error=0.05, error_type=ErrorType.TYPE_I
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
