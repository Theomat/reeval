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

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.02
)
text_length = Measure(
    "length", MeasureType.MEAN, absolute_error=10, value_range=(1, 1000)
)


my_eval = Evaluation([ai_generated, text_length], confidence=0.95, max_comparisons=1)
print(my_eval.compute_sample_size())
```

### Confidence

Here given the two measures and the sample size, the objective is to compute the actual confidence that all results are true at the same time and each individually.

```python
from reeval import Evaluation, Measure, MeasureType

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.02
)
text_length = Measure(
    "length", MeasureType.MEAN, absolute_error=10, value_range=(1, 1000)
)


my_eval = Evaluation([ai_generated, text_length], sample_size=300, max_comparisons=1)
print(my_eval.compute_sample_size())
```

### Absolute Error

Here given the two measures, the sample size and the confidence, the objective is to compute the actual absolute error across measures.

```python
from reeval import Evaluation, Measure, MeasureType

ai_generated = Measure(
    "ai_generated", MeasureType.PROPORTION_BOOLEAN
)
text_length = Measure(
    "length", MeasureType.MEAN, value_range=(1, 1000)
)


my_eval = Evaluation(
    [ai_generated, text_length], confidence=0.95, sample_size=300, max_comparisons=1
)
print(my_eval.compute_absolute_errors())
```
