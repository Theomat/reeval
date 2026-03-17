from dataclasses import dataclass, field
import math
from reeval.error_control import AlphaErrorControl, ErrorControl
from reeval.measures.measure import (
    Measure,
    apply_bonferroni,
    normal_cdf,
    normal_min_detectable_effect,
    normal_power,
    normal_power_sample_size,
    normal_sample_size,
    normal_z,
    reverse_bonferroni,
)
from scipy import stats


__all__ = ["RankMeasure"]


@dataclass(eq=False)
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
        return math.sqrt((self.max_rank**2 - 1) / 12)

    def compute_sample_size(
        self,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            return normal_sample_size(alpha, self.std, self.absolute_error)
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        beta = apply_bonferroni(
            error_control.beta, self.repeats * repetition_multiplier
        )
        return normal_power_sample_size(alpha, beta, self.std, self.absolute_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            z = normal_z(alpha) * self.std
        else:
            alpha = apply_bonferroni(
                error_control.significance_level, self.repeats * repetition_multiplier
            )
            beta = apply_bonferroni(
                error_control.beta, self.repeats * repetition_multiplier
            )
            return normal_min_detectable_effect(alpha, beta, self.std, sample_size)
        return z / math.sqrt(sample_size)

    def compute_error_probability(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        adjusted_sample_size = math.sqrt(sample_size)
        if isinstance(error_control, AlphaErrorControl):
            tail_probability = 2 * (
                1 - normal_cdf(adjusted_sample_size * self.absolute_error / self.std)
            )
            alpha = reverse_bonferroni(
                tail_probability, self.repeats * repetition_multiplier
            )
            alpha = min(max(alpha, 0.0), 1.0)
            return 1 - alpha
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        raw_power = normal_power(
            adjusted_sample_size * self.absolute_error / self.std, alpha
        )
        beta = reverse_bonferroni(
            1 - raw_power,
            self.repeats * repetition_multiplier,
        )
        beta = min(max(beta, 0.0), 1.0)
        return 1 - beta

    def test_different(
        self,
        sample1: list[float],
        sample2: list[float],
        error_control: ErrorControl = ErrorControl.type_i(0.05),
    ) -> tuple[float, float, tuple[float, float], float, float]:
        """Applies a two-tailed test for two samples of ranked data.
        It checks if the rank distributions are the same.
        It relies on the Mann-Whitney U test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error_control (ErrorControl): alpha control gives a two-sided CI;
                power control gives the narrower beta-calibrated interval.

        Returns:
            float: the p-value obtained
            float: effect size (Vargha and Delaney's A12)
            tuple[float, float]: confidence interval of A12
            float: Type I error (α) for the given sample size
            float: Type II error (β) for the given sample size
        """
        n1, n2 = len(sample1), len(sample2)
        u_result = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        p_value = u_result.pvalue
        a12 = u_result.statistic / (n1 * n2)

        # Normal approximation CI for A12
        se = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        if isinstance(error_control, AlphaErrorControl):
            z = stats.norm.ppf(1 - error_control.alpha / 2)
            type_i_control = error_control
            type_ii_control = ErrorControl.type_ii(
                error_control.alpha, significance_level=error_control.alpha
            )
        else:
            z = stats.norm.ppf(1 - error_control.beta)
            type_i_control = ErrorControl.type_i(error_control.significance_level)
            type_ii_control = error_control
        ci = (max(0.0, a12 - z * se), min(1.0, a12 + z * se))

        n = min(n1, n2)
        type_i_error = 1 - self.compute_error_probability(n, type_i_control)
        type_ii_error = 1 - self.compute_error_probability(n, type_ii_control)
        return p_value, a12, ci, type_i_error, type_ii_error
