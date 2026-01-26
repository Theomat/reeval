# Reliable and Efficient EVALuations

The goal of this python package is to provide a framework in order to give theoretical guarantees for evaluations.
In order to do so, evaluation creation is viewed as sampling randomly instances of the task.
Therefore, given a confidence level and an absolute error it is easy to compute a sample size; meaning here that the evaluation must contain at least this number of instances.

## Examples

An evaluation can contain multiple measures on the same data or on different data.

### Sample size

Here given the two measures, the objective is to know the minimum number of instances required for the target evaluation.

```python
from reeval import Evaluation, Measure, MeasureType
from reeval.population import InfinitePopulation

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.02
)
text_length = Measure(
    "length", MeasureType.MEAN, absolute_error=10, value_range=(1, 1000)
)


my_eval = Evaluation(
    [ai_generated, text_length],
    confidence=0.95,
    population=InfinitePopulation(),
)
print(my_eval.compute_sample_size())
```

### Confidence

Here given the two measures and the sample size, the objective is to compute the actual confidence that all results are true at the same time and each individually.

```python
from reeval import Evaluation, Measure, MeasureType
from reeval.population import InfinitePopulation

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.02
)
text_length = Measure(
    "length", MeasureType.MEAN, absolute_error=10, value_range=(1, 1000)
)


my_eval = Evaluation(
    [ai_generated, text_length],
    sample_size=300,
    population=InfinitePopulation(),
)
print(my_eval.compute_confidences())
```

### Absolute Error

Here given the two measures, the sample size and the confidence, the objective is to compute the actual absolute error across measures.

```python
from reeval import Evaluation, Measure, MeasureType
from reeval.population import InfinitePopulation

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN
)
text_length = Measure(
    "length", MeasureType.MEAN, value_range=(1, 1000)
)


my_eval = Evaluation(
    [ai_generated, text_length],
    confidence=0.95,
    sample_size=300,
    population=InfinitePopulation(),
)
print(my_eval.compute_absolute_errors())
```

### Finite Population

When sampling from a finite population, sample size requirements are reduced using Cochran's formula.

```python
from reeval import Evaluation, Measure, MeasureType
from reeval.population import FinitePopulation

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.02
)

my_eval = Evaluation(
    [ai_generated],
    confidence=0.95,
    population=FinitePopulation(size=10000),
)
print(my_eval.compute_sample_size())
```

### Filtered Population

When evaluating a subset of the population (e.g., only instances where a certain condition is true), use `FilteredPopulation` to account for the dependency between evaluations.

```python
from reeval import Evaluation, Measure, MeasureType, compute_global_sample_sizes
from reeval.population import FinitePopulation

# First evaluation on the full population
ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.02, empirical_value=0.3
)
full_eval = Evaluation(
    [ai_generated],
    confidence=0.99,
    population=FinitePopulation(size=10000),
)

# Second evaluation only on AI-generated instances
quality = Measure("quality", MeasureType.MEAN, absolute_error=0.1, value_range=(0, 10))
filtered_pop = full_eval.population.filter_on(full_eval, ai_generated)
filtered_eval = Evaluation(
    [quality],
    confidence=0.95,
    population=filtered_pop,
)

# Compute sample sizes for all evaluations together
sample_sizes = compute_global_sample_sizes([filtered_eval])
print(sample_sizes)
```
