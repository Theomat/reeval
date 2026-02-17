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


__all__ = ["VarianceMeasure"]


@dataclass(eq=False)
class VarianceMeasure(Measure):
    relative_error: float | None = field(default=None)
    """Relative error of the measure.
    """

    def compute_sample_size(self, confidence: float, repetition_multiplier: int = 1):
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        return normal_sample_size(alpha, 1, self.relative_error)

    def compute_absolute_error(
        self, sample_size: int, confidence: float, repetition_multiplier: int = 1
    ):
        raise NotImplementedError(
            "cannot compute an absolute error for the variance measure"
        )

    def compute_relative_error(
        self, sample_size: int, confidence: float, repetition_multiplier: int = 1
    ):
        """Compute relative error of the measure.

        Args:
            sample_size (int): sample size used
            confidence (float): [0; 1]

        Returns:
            float: relative error
        """
        alpha = 1 - confidence
        alpha = apply_bonferroni(alpha, self.repeats * repetition_multiplier)
        z = normal_z(alpha)
        return z / math.sqrt(sample_size)

    def compute_confidence(self, sample_size: int, repetition_multiplier: int = 1):
        adjusted_sample_size = math.sqrt(sample_size)
        confidence = normal_cdf(adjusted_sample_size * self.relative_error)
        alpha = reverse_bonferroni(1 - confidence, self.repeats * repetition_multiplier)
        return 1 - alpha
