from dataclasses import dataclass, field
import math
from reeval.error_type import ErrorType
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
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        match error_type:
            case ErrorType.TYPE_I:
                # Controls false positive rate α using two-sided normal quantile z_{α/2}
                # n = (z_{α/2} · σ / δ)²
                alpha = apply_bonferroni(error, self.repeats * repetition_multiplier)
                return normal_sample_size(alpha, self.std, self.absolute_error)
            case ErrorType.TYPE_II:
                # Power analysis: minimum n to achieve power 1-β
                # n = (z_β · σ / δ)² where z_β = Φ⁻¹(1-β), one-sided quantile
                beta = apply_bonferroni(error, self.repeats * repetition_multiplier)
                return normal_sample_size(2 * beta, self.std, self.absolute_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        match error_type:
            case ErrorType.TYPE_I:
                # Two-sided CI half-width at (1-α) level: z_{α/2} · σ / √n
                alpha = apply_bonferroni(error, self.repeats * repetition_multiplier)
                z = normal_z(alpha) * self.std
            case ErrorType.TYPE_II:
                # Minimum detectable effect at power 1-β: z_β · σ / √n
                # z_β = Φ⁻¹(1-β), one-sided power quantile
                beta = apply_bonferroni(error, self.repeats * repetition_multiplier)
                z = normal_z(2 * beta) * self.std
        return z / math.sqrt(sample_size)

    def compute_error_probability(
        self,
        sample_size: int,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        adjusted_sample_size = math.sqrt(sample_size)
        match error_type:
            case ErrorType.TYPE_I:
                # Confidence level 1 - α: Φ(√n · δ/σ) = 1 - α/2, invert for α
                confidence = normal_cdf(
                    adjusted_sample_size * self.absolute_error / self.std
                )
                alpha = reverse_bonferroni(
                    1 - confidence, self.repeats * repetition_multiplier
                )
                return 1 - alpha
            case ErrorType.TYPE_II:
                # Power 1 - β: Φ(√n · δ/σ) = 1 - β (one-sided), invert for β
                confidence = normal_cdf(
                    adjusted_sample_size * self.absolute_error / self.std
                )
                beta = reverse_bonferroni(
                    1 - confidence, self.repeats * repetition_multiplier
                )
                return 1 - beta

    def test_different(
        self,
        sample1: list[float],
        sample2: list[float],
        error: float = 0.05,
        error_type: ErrorType = ErrorType.TYPE_I,
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of ranked data.
        It checks if the rank distributions are the same.
        It relies on the Mann-Whitney U test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error (float): error rate for the CI; interpreted as α (TYPE_I) or β (TYPE_II)
            error_type (ErrorType): TYPE_I uses two-sided CI at (1-α) level;
                TYPE_II uses one-sided power CI at (1-β) level (normal approximation for A12)

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
        se = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        match error_type:
            case ErrorType.TYPE_I:
                # Two-sided CI at (1-α) level: z_{α/2} = Φ⁻¹(1-α/2)
                z = stats.norm.ppf(1 - error / 2)
            case ErrorType.TYPE_II:
                # Power-focused CI at (1-β) level: one-sided z_β = Φ⁻¹(1-β)
                z = stats.norm.ppf(1 - error)
        ci = (max(0.0, a12 - z * se), min(1.0, a12 + z * se))

        return p_value, a12, ci
