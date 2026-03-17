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


__all__ = ["VarianceMeasure"]


@dataclass(eq=False)
class VarianceMeasure(Measure):
    relative_error: float | None = field(default=None)
    """Relative error of the measure.
    """

    def compute_sample_size(
        self,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            return normal_sample_size(alpha, 1, self.relative_error)
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        beta = apply_bonferroni(
            error_control.beta, self.repeats * repetition_multiplier
        )
        return normal_power_sample_size(alpha, beta, 1, self.relative_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        raise NotImplementedError(
            "cannot compute an absolute error for the variance measure"
        )

    def compute_relative_error(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        """Compute relative error of the measure.

        Args:
            sample_size (int): sample size used
            error_control (ErrorControl): alpha control gives two-sided relative error;
                power control gives the minimum detectable relative effect at the
                requested beta and significance level.

        Returns:
            float: relative error
        """
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            z = normal_z(alpha)
        else:
            alpha = apply_bonferroni(
                error_control.significance_level, self.repeats * repetition_multiplier
            )
            beta = apply_bonferroni(
                error_control.beta, self.repeats * repetition_multiplier
            )
            return normal_min_detectable_effect(alpha, beta, 1, sample_size)
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
                1 - normal_cdf(adjusted_sample_size * self.relative_error)
            )
            alpha = reverse_bonferroni(
                tail_probability, self.repeats * repetition_multiplier
            )
            alpha = min(max(alpha, 0.0), 1.0)
            return 1 - alpha
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        raw_power = normal_power(adjusted_sample_size * self.relative_error, alpha)
        beta = reverse_bonferroni(
            1 - raw_power,
            self.repeats * repetition_multiplier,
        )
        beta = min(max(beta, 0.0), 1.0)
        return 1 - beta

    def test_different(
        self,
        sample1: list[bool],
        sample2: list[bool],
        error_control: ErrorControl = ErrorControl.type_i(0.05),
    ) -> tuple[float, float, tuple[float, float], float, float]:
        """Not implemented."""
        raise NotImplementedError()
