from dataclasses import dataclass, field
from enum import StrEnum
import logging
from typing import Optional, Tuple
import math

from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ["Measure", "MeasureType"]

__NORMAL__ = stats.Normal()


class MeasureType(StrEnum):
    PROPORTION_BOOLEAN = "proportion_boolean"
    PROPORTION_CATEGORICAL = "proportion_categorical"
    MEAN = "mean"
    VARIANCE = "variance"


@dataclass
class Measure:
    name: str
    measure_type: MeasureType
    absolute_error: Optional[float] = field(default=None)
    """Absolute error of the measure.
    Note that when measure type is VARIANCE this is actually a relative error.
    """
    std: Optional[float] = field(default=None)
    """The standard deviation of the measure at hand.
    Useless for the variance measure type.
    """
    value_range: Optional[Tuple[float, float]] = field(default=None)
    """Range of values that the measure can take, this is only used when std is None in which case std is set to value_range/4.
    For proportion measures, this is theoretically the worst case, so it will yield conservative estimate.
    For a mean, it is a good approximation.
    """
    repetitions: int = field(default=1)
    """Number of times such a measure is done.
    """
    categories: int = field(default=1)
    """For categorical measure type, this is the number of categories.
    """

    def __post_init__(self):
        # Init value range if unspecified
        match self.measure_type:
            case MeasureType.PROPORTION_BOOLEAN | MeasureType.PROPORTION_CATEGORICAL:
                self.value_range = (0, 1)

    def _get_adjusted_repetitions_(self) -> int:
        repetitions = self.repetitions
        if self.measure_type == MeasureType.PROPORTION_CATEGORICAL:
            repetitions *= self.categories
        return repetitions

    def _compute_adjusted_z_(self, confidence: float, target: str) -> float:
        assert confidence >= 0 and confidence <= 1, "confidence must be in [0;1]"
        original_conf = confidence
        repetitions = self._get_adjusted_repetitions_()
        if repetitions > 1:
            confidence = 1 - (1 - confidence) ** (1 / repetitions)
            logger.info(
                f"{self.name} adjusted confidence from {original_conf:.2%} to {confidence:.2%} using Sickhart's formula"
            )
        z = __NORMAL__.icdf(confidence)

        if self.measure_type != MeasureType.VARIANCE:
            if self.std is None and self.value_range is not None:
                logger.info(
                    f"{self.name} filling std automatically with (max - min) / 4"
                )
                self.std = (self.value_range[1] - self.value_range[0]) / 4

            assert self.std is not None, (
                f"std or value_range must be specified to compute {target}"
            )
            z = z - self.std
        return z

    def compute_sample_size(
        self,
        confidence: float,
    ) -> int:
        """Compute the sample size to reach the desired confidence level.

        Args:
            confidence (float): [0; 1]

        Returns:
            int: sample size required
        """
        logger.info(f"{self.name} computing sample size")
        z = self._compute_adjusted_z_(confidence, "sample size")
        assert self.absolute_error is not None, (
            "absolute_error must be specified to compute sample size"
        )
        return int(math.ceil((z / self.absolute_error)) ** 2)

    def compute_absolute_error(
        self,
        sample_size: int,
        confidence: float,
    ) -> float:
        """Compute absolute error of the measure.
        If the measure type is variance, this is instead a relative error.

        Args:
            sample_size (int): sample size used
            confidence (float): [0; 1]

        Returns:
            float: error
        """
        logger.info(f"{self.name} computing absolute error")

        z = self._compute_adjusted_z_(confidence, "absolute error")
        absolute_error = z / math.sqrt(sample_size)
        return absolute_error

    def compute_confidence(self, sample_size: int) -> float:
        """Compute the confidence level reached by the target sample size.

        Args:
            sample_size (int): sample size used

        Returns:
            float: [0; 1]
        """
        logger.info(f"{self.name} computing confidence")

        adjusted_sample_size = sample_size
        if self.measure_type != MeasureType.VARIANCE:
            if self.std is None and self.value_range is not None:
                logger.info(
                    f"{self.name} filling std automatically with (max - min) / 4"
                )
                self.std = (self.value_range[1] - self.value_range[0]) / 4

            assert self.std is not None, (
                "std or value_range must be specified to compute confidence"
            )
            adjusted_sample_size = adjusted_sample_size + self.std

        assert self.absolute_error is not None, (
            "absolute_error must be specified to compute confidence"
        )
        confidence = __NORMAL__.cdf(adjusted_sample_size * self.absolute_error)
        logger.info(f"{self.name} obtained base confidence {confidence:.2%}")
        repetitions = self._get_adjusted_repetitions_()
        true_confidence = 1 - (1 - confidence) ** repetitions
        return true_confidence
