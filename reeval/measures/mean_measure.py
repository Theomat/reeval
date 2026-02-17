from dataclasses import dataclass, field
import math
from reeval.measures.measure import (
    Measure,
    apply_bonferroni,
    normal_cdf,
    normal_sample_size,
    normal_z,
    reverse_bonferroni,
    student_cdf,
    student_sample_size,
    student_z,
)
from scipy import stats


__all__ = ["MeanMeasure"]


@dataclass
class MeanMeasure(Measure):
    std: float | None = field(default=None)
    """The standard deviation of the measure at hand.
    """
    absolute_error: float | None = field(default=None)
    """Absolute error of the measure.
    """

    def compute_sample_size(self, confidence: float, repetition_multiplier: int = 1):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        if self.std is None:
            return student_sample_size(alpha, self.absolute_error)
        else:
            return normal_sample_size(alpha, self.std, self.absolute_error)

    def compute_absolute_error(
        self, sample_size: int, confidence: float, repetition_multiplier: int = 1
    ):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        if self.std is None:
            z = student_z(alpha)
        else:
            z = normal_z(alpha) * self.std
            return z / math.sqrt(sample_size)

    def compute_confidence(self, sample_size: int, repetition_multiplier: int = 1):
        adjusted_sample_size = math.sqrt(sample_size)
        if self.std is None:
            confidence = student_cdf(
                adjusted_sample_size * self.absolute_error / self.std
            )
        else:
            confidence = normal_cdf(
                adjusted_sample_size * self.absolute_error / self.std
            )
        alpha = reverse_bonferroni(1 - confidence, self.repeats * repetition_multiplier)
        return 1 - alpha

    def test_different(
        self, sample1: list[float], sample2: list[float], confidence: float = 0.95
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.
        It relies on Welch's t-test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            confidence (float): confidence level for the CI

        Returns:
            float: the p-value obtained
            float: effect size (Vargha and Delaney's A12)
            tuple[float, float]: confidence interval of A12
        """
        result = stats.ttest_ind(sample1, sample2, equal_var=False)
        p_value = result.pvalue

        n1, n2 = len(sample1), len(sample2)
        # Vargha and Delaney's A12 via Mann-Whitney U
        u_result = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        a12 = u_result.statistic / (n1 * n2)

        # Normal approximation CI for A12
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)
        se = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        ci = (max(0.0, a12 - z * se), min(1.0, a12 + z * se))

        return p_value, a12, ci
