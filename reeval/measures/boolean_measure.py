from dataclasses import dataclass, field
import logging
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
from reeval.population import FilteredPopulation, Population
from scipy import stats

logger = logging.getLogger(__name__)
__all__ = ["BooleanMeasure", "CategoricalMeasures"]


__DEFAULT_STD__ = 0.5


@dataclass(eq=False)
class BooleanMeasure(Measure):
    std: float | None = field(default=None)
    """The standard deviation of the measure at hand.
    """
    absolute_error: float | None = field(default=None)
    """Absolute error of the measure.
    """

    def compute_sample_size(
        self,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        std = __DEFAULT_STD__ if self.std is None else self.std
        match error_type:
            case ErrorType.TYPE_I:
                # Controls false positive rate α using two-sided normal quantile z_{α/2}
                # n = (z_{α/2} · σ / δ)²
                alpha = apply_bonferroni(error, self.repeats * repetition_multiplier)
                return normal_sample_size(alpha, std, self.absolute_error)
            case ErrorType.TYPE_II:
                # Power analysis: minimum n to achieve power 1-β
                # n = (z_β · σ / δ)² where z_β = Φ⁻¹(1-β), using one-sided quantile
                beta = apply_bonferroni(error, self.repeats * repetition_multiplier)
                return normal_sample_size(2 * beta, std, self.absolute_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        std = __DEFAULT_STD__ if self.std is None else self.std
        match error_type:
            case ErrorType.TYPE_I:
                # Two-sided CI half-width at (1-α) level: z_{α/2} · σ / √n
                alpha = apply_bonferroni(error, self.repeats * repetition_multiplier)
                z = normal_z(alpha) * std
            case ErrorType.TYPE_II:
                # Minimum detectable effect at power 1-β: z_β · σ / √n
                # z_β = Φ⁻¹(1-β), one-sided power quantile
                beta = apply_bonferroni(error, self.repeats * repetition_multiplier)
                z = normal_z(2 * beta) * std
        return z / math.sqrt(sample_size)

    def compute_error_probability(
        self,
        sample_size: int,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        adjusted_sample_size = math.sqrt(sample_size)
        std = __DEFAULT_STD__ if self.std is None else self.std
        match error_type:
            case ErrorType.TYPE_I:
                # Confidence level 1 - α: Φ(√n · δ/σ) = 1 - α/2, invert for α
                confidence = normal_cdf(
                    adjusted_sample_size * self.absolute_error / std
                )
                alpha = reverse_bonferroni(
                    1 - confidence, self.repeats * repetition_multiplier
                )
                return 1 - alpha
            case ErrorType.TYPE_II:
                # Power 1 - β: Φ(√n · δ/σ) = 1 - β (one-sided), invert for β
                confidence = normal_cdf(
                    adjusted_sample_size * self.absolute_error / std
                )
                beta = reverse_bonferroni(
                    1 - confidence, self.repeats * repetition_multiplier
                )
                return 1 - beta

    def filter(
        self,
        population: Population,
        empirical_value: float,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
    ) -> FilteredPopulation:
        """
        Filter a population based on the result of this measure.
        """
        match error_type:
            case ErrorType.TYPE_I:
                # Confidence threshold: (1-α) level
                confidence = 1 - error
            case ErrorType.TYPE_II:
                # Power threshold: (1-β) used as confidence criterion
                confidence = 1 - error
        return FilteredPopulation(
            population, confidence, empirical_value, self.absolute_error
        )

    def test_different(
        self,
        sample1: list[bool],
        sample2: list[bool],
        error: float = 0.05,
        error_type: ErrorType = ErrorType.TYPE_I,
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.
        It relies on Fisher's exact test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error (float): error rate for the odds ratio CI; interpreted as α (TYPE_I)
                or β (TYPE_II)
            error_type (ErrorType): TYPE_I uses two-sided CI at (1-α) level;
                TYPE_II uses one-sided power CI at (1-β) level (Woolf logit method)

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
        match error_type:
            case ErrorType.TYPE_I:
                # Two-sided CI at (1-α) level: z_{α/2} = Φ⁻¹(1-α/2)
                z = -stats.norm.ppf(error / 2)
            case ErrorType.TYPE_II:
                # Power-focused CI at (1-β) level: one-sided z_β = Φ⁻¹(1-β)
                z = -stats.norm.ppf(error)
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
