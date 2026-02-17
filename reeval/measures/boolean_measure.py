from dataclasses import dataclass, field
import logging
import math
from reeval.measures.measure import (
    Measure,
    apply_bonferroni,
    normal_cdf,
    normal_sample_size,
    normal_z,
    reverse_bonferroni,
)
from reeval.population import FilteredPopulation, Population
from scipy import stats

logger = logging.getLogger(__name__)
__all__ = ["BooleanMeasure", "CategoricalMeasures"]


__DEFAULT_STD__ = 0.5**2


@dataclass(eq=False)
class BooleanMeasure(Measure):
    std: float | None = field(default=None)
    """The standard deviation of the measure at hand.
    """
    absolute_error: float | None = field(default=None)
    """Absolute error of the measure.
    """

    def compute_sample_size(self, confidence: float, repetition_multiplier: int = 1):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        std = __DEFAULT_STD__ if self.std is None else self.std
        return normal_sample_size(alpha, std, self.absolute_error)

    def compute_absolute_error(
        self, sample_size: int, confidence: float, repetition_multiplier: int = 1
    ):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        std = __DEFAULT_STD__ if self.std is None else self.std
        z = normal_z(alpha) * std
        return z / math.sqrt(sample_size)

    def compute_confidence(self, sample_size: int, repetition_multiplier: int = 1):
        adjusted_sample_size = math.sqrt(sample_size)
        std = __DEFAULT_STD__ if self.std is None else self.std
        confidence = normal_cdf(adjusted_sample_size * self.absolute_error / std)
        alpha = reverse_bonferroni(1 - confidence, self.repeats * repetition_multiplier)
        return 1 - alpha

    def filter(
        self, population: Population, empirical_value: float, confidence: float
    ) -> FilteredPopulation:
        """
        Filter a population based on the result of this measure.
        """
        return FilteredPopulation(
            population, confidence, empirical_value, self.absolute_error
        )

    def test_different(
        self, sample1: list[bool], sample2: list[bool], confidence: float = 0.95
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.
        It relies on Fisher's exact test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            confidence (float): confidence level for the odds ratio CI

        Returns:
            float: the p-value obtained
            float: effect size (odds ratio)
            tuple[float, float]: confidence interval of the odds ratio
        """
        s1, s2 = sum(sample1), sum(sample2)
        f1, f2 = len(sample1) - s1, len(sample2) - s2
        table = [[s1, f1], [s2, f2]]
        result = stats.fisher_exact(table)
        odds_ratio = result.statistic
        # Woolf logit method for CI: SE(log(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
        alpha = 1 - confidence
        z = -stats.norm.ppf(alpha / 2)
        if 0 in (s1, f1, s2, f2):
            ci = (0.0, math.inf)
        else:
            log_or = math.log(odds_ratio)
            se = math.sqrt(1 / s1 + 1 / f1 + 1 / s2 + 1 / f2)
            ci = (math.exp(log_or - z * se), math.exp(log_or + z * se))
        return result.pvalue, odds_ratio, ci


def CategoricalMeasures(
    name: str,
    categories: int,
    std: float | None = None,
    absolute_error: float | None = None,
    repeats: int = 1,
) -> list[BooleanMeasure]:
    return [
        BooleanMeasure(f"{name}_{i}", repeats, std, absolute_error)
        for i in range(categories)
    ]
