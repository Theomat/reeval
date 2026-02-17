from dataclasses import dataclass, field
import math
from reeval.measures.measure import (
    Measure,
    apply_bonferroni,
    normal_cdf,
    normal_sample_size,
    normal_z,
    reverse_bonferroni,
)
from scipy import stats


__all__ = ["RankMeasure"]


@dataclass
class RankMeasure(Measure):
    max_rank: int = field(default=2)
    """The number of items being ranked (rank values in 1..k).
    """
    absolute_error: float | None = field(default=None)
    """Absolute error of the measure.
    """

    @property
    def std(self) -> float:
        """Standard deviation of a discrete uniform distribution over {1, ..., k}."""
        return math.sqrt((self.k**2 - 1) / 12)

    def compute_sample_size(self, confidence: float, repetition_multiplier: int = 1):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        return normal_sample_size(alpha, self.std, self.absolute_error)

    def compute_absolute_error(
        self, sample_size: int, confidence: float, repetition_multiplier: int = 1
    ):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        z = normal_z(alpha) * self.std
        return z / math.sqrt(sample_size)

    def compute_confidence(self, sample_size: int, repetition_multiplier: int = 1):
        adjusted_sample_size = math.sqrt(sample_size)
        confidence = normal_cdf(adjusted_sample_size * self.absolute_error / self.std)
        alpha = reverse_bonferroni(1 - confidence, self.repeats * repetition_multiplier)
        return 1 - alpha

    def test_different(
        self, sample1: list[float], sample2: list[float], confidence: float = 0.95
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of ranked data.
        It checks if the rank distributions are the same.
        It relies on the Mann-Whitney U test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            confidence (float): confidence level for the CI

        Returns:
            float: the p-value obtained
            float: effect size (Vargha and Delaney's A12)
            tuple[float, float]: confidence interval of A12
        """
        n1, n2 = len(sample1), len(sample2)
        u_result = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        p_value = u_result.pvalue
        a12 = u_result.statistic / (n1 * n2)

        # Normal approximation CI for A12
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)
        se = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        ci = (max(0.0, a12 - z * se), min(1.0, a12 + z * se))

        return p_value, a12, ci
