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


__all__ = ["VarianceMeasure"]


@dataclass(eq=False)
class VarianceMeasure(Measure):
    relative_error: float | None = field(default=None)
    """Relative error of the measure.
    """

    def compute_sample_size(
        self,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        match error_type:
            case ErrorType.TYPE_I:
                # Controls false positive rate α using two-sided normal quantile z_{α/2}
                # n = (z_{α/2} / δ_rel)²  (unit std for relative error)
                alpha = apply_bonferroni(error, self.repeats * repetition_multiplier)
                return normal_sample_size(alpha, 1, self.relative_error)
            case ErrorType.TYPE_II:
                # Power analysis: minimum n to achieve power 1-β
                # n = (z_β / δ_rel)² where z_β = Φ⁻¹(1-β), one-sided quantile
                beta = apply_bonferroni(error, self.repeats * repetition_multiplier)
                return normal_sample_size(2 * beta, 1, self.relative_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        raise NotImplementedError(
            "cannot compute an absolute error for the variance measure"
        )

    def compute_relative_error(
        self,
        sample_size: int,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ):
        """Compute relative error of the measure.

        Args:
            sample_size (int): sample size used
            error (float): error rate in [0; 1]; interpreted as α (TYPE_I) or β (TYPE_II)
            error_type (ErrorType): TYPE_I gives two-sided relative error at (1-α) level;
                TYPE_II gives minimum detectable relative effect at power 1-β
                (z_β / √n where z_β = Φ⁻¹(1-β), one-sided quantile)

        Returns:
            float: relative error
        """
        match error_type:
            case ErrorType.TYPE_I:
                # Two-sided relative error at (1-α) level: z_{α/2} / √n
                alpha = apply_bonferroni(error, self.repeats * repetition_multiplier)
                z = normal_z(alpha)
            case ErrorType.TYPE_II:
                # Minimum detectable relative effect at power 1-β: z_β / √n
                # z_β = Φ⁻¹(1-β), one-sided power quantile
                beta = apply_bonferroni(error, self.repeats * repetition_multiplier)
                z = normal_z(2 * beta)
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
                # Confidence level 1 - α: Φ(√n · δ_rel) = 1 - α/2, invert for α
                confidence = normal_cdf(adjusted_sample_size * self.relative_error)
                alpha = reverse_bonferroni(
                    1 - confidence, self.repeats * repetition_multiplier
                )
                return 1 - alpha
            case ErrorType.TYPE_II:
                # Power 1 - β: Φ(√n · δ_rel) = 1 - β (one-sided), invert for β
                confidence = normal_cdf(adjusted_sample_size * self.relative_error)
                beta = reverse_bonferroni(
                    1 - confidence, self.repeats * repetition_multiplier
                )
                return 1 - beta

    def test_different(
        self,
        sample1: list[bool],
        sample2: list[bool],
        error: float = 0.05,
        error_type: ErrorType = ErrorType.TYPE_I,
    ) -> tuple[float, float, tuple[float, float]]:
        """Not implemented."""
        raise NotImplementedError()
