from dataclasses import dataclass, field
import logging
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
    sensitivity: float = field(default=1.0)
    """Sensitivity of the labels being sampled, i.e. P(observed=1 | true=1).
    """
    specificity: float = field(default=1.0)
    """Specificity of the labels being sampled, i.e. P(observed=0 | true=0).
    """

    def _effective_std(self) -> float:
        std = __DEFAULT_STD__ if self.std is None else self.std
        correction = self.sensitivity + self.specificity - 1
        if correction <= 0:
            raise ValueError(
                "sensitivity + specificity must be strictly greater than 1."
            )
        # Standard prevalence correction for imperfect binary labels:
        # p = (q + Sp - 1) / (Se + Sp - 1),
        # where p is the true Bernoulli rate, q the observed positive rate,
        # Se sensitivity, and Sp specificity.
        #
        # This is an affine transform of q, so:
        # Var[p] = Var[q] / (Se + Sp - 1)^2
        # and therefore std[p] = std[q] / |Se + Sp - 1|.
        return std / correction

    def compute_sample_size(
        self,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        std = self._effective_std()
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            return normal_sample_size(alpha, std, self.absolute_error)
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        beta = apply_bonferroni(
            error_control.beta, self.repeats * repetition_multiplier
        )
        return normal_power_sample_size(alpha, beta, std, self.absolute_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        std = self._effective_std()
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            z = normal_z(alpha) * std
        else:
            alpha = apply_bonferroni(
                error_control.significance_level, self.repeats * repetition_multiplier
            )
            beta = apply_bonferroni(
                error_control.beta, self.repeats * repetition_multiplier
            )
            return normal_min_detectable_effect(alpha, beta, std, sample_size)
        return z / math.sqrt(sample_size)

    def compute_error_probability(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        adjusted_sample_size = math.sqrt(sample_size)
        std = self._effective_std()
        if isinstance(error_control, AlphaErrorControl):
            tail_probability = 2 * (
                1 - normal_cdf(adjusted_sample_size * self.absolute_error / std)
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
            adjusted_sample_size * self.absolute_error / std, alpha
        )
        beta = reverse_bonferroni(
            1 - raw_power,
            self.repeats * repetition_multiplier,
        )
        beta = min(max(beta, 0.0), 1.0)
        return 1 - beta

    def test_different(
        self,
        sample1: list[bool | int],
        sample2: list[bool | int],
        error_control: ErrorControl = ErrorControl.type_i(0.05),
    ) -> tuple[float, float, tuple[float, float], float, float]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.
        It relies on Fisher's exact test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error_control (ErrorControl): alpha control gives a two-sided
                odds-ratio interval; power control gives the narrower
                beta-calibrated interval.

        Returns:
            float: the p-value obtained
            float: effect size (odds ratio)
            tuple[float, float]: confidence interval of the odds ratio
            float: Type I error (α) for the given sample size
            float: Type II error (β) for the given sample size
        """

        def to_binary(value: bool | int) -> int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int) and value in (0, 1):
                return value
            raise ValueError(
                "BooleanMeasure.test_different expects binary values (bool or 0/1 int)."
            )

        correction = self.sensitivity + self.specificity - 1
        if correction <= 0:
            raise ValueError(
                "sensitivity + specificity must be strictly greater than 1."
            )

        def adjust_successes(sample: list[bool | int]) -> int:
            n = len(sample)
            observed_successes = sum(to_binary(v) for v in sample)
            observed_rate = observed_successes / n
            adjusted_rate = (observed_rate + self.specificity - 1) / correction
            adjusted_rate = min(1.0, max(0.0, adjusted_rate))
            return int(round(adjusted_rate * n))

        s1, s2 = adjust_successes(sample1), adjust_successes(sample2)
        f1, f2 = len(sample1) - s1, len(sample2) - s2
        table = [[s1, f1], [s2, f2]]
        result = stats.fisher_exact(table)
        odds_ratio = result.statistic
        # Woolf logit method for CI: SE(log(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
        if isinstance(error_control, AlphaErrorControl):
            z = -stats.norm.ppf(error_control.alpha / 2)
            type_ii_control = ErrorControl.type_ii(
                error_control.alpha, significance_level=error_control.alpha
            )
            type_i_control = error_control
        else:
            z = -stats.norm.ppf(error_control.beta)
            type_ii_control = error_control
            type_i_control = ErrorControl.type_i(error_control.significance_level)
        if 0 in (s1, f1, s2, f2):
            ci = (0.0, math.inf)
        else:
            log_or = math.log(odds_ratio)
            se = math.sqrt(1 / s1 + 1 / f1 + 1 / s2 + 1 / f2)
            ci = (math.exp(log_or - z * se), math.exp(log_or + z * se))
        n = min(len(sample1), len(sample2))
        type_i_error = 1 - self.compute_error_probability(n, type_i_control)
        type_ii_error = 1 - self.compute_error_probability(n, type_ii_control)
        return result.pvalue, odds_ratio, ci, type_i_error, type_ii_error


def CategoricalMeasures(
    name: str,
    categories: int,
    std: float | None = None,
    absolute_error: float | None = None,
    sensitivity: float = 1,
    specificity: float = 1,
    repeats: int = 1,
) -> list[BooleanMeasure]:
    return [
        BooleanMeasure(
            name=f"{name}_{i}",
            repeats=repeats,
            std=std,
            absolute_error=absolute_error,
            sensitivity=sensitivity,
            specificity=specificity,
        )
        for i in range(categories)
    ]
