from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import math

from scipy import stats

from reeval.error_type import ErrorType

logger = logging.getLogger(__name__)

__all__ = [
    "Measure",
    "ErrorType",
    "apply_bonferroni",
    "reverse_bonferroni",
    "normal_sample_size",
    "normal_z",
    "normal_cdf",
    "student_sample_size",
    "student_cdf",
    "student_z",
]
__NORMAL__ = stats.Normal()
__STUDENT__ = stats.t


def __student_icdf__(alpha: float, sample_size: int) -> float:
    return __STUDENT__.isf(alpha, df=sample_size - 1)


def apply_bonferroni(alpha: float, repetitions: int) -> float:
    logging.debug(f"bonferroni -> (alpha={alpha} repetitions={repetitions})")
    return alpha / repetitions


def reverse_bonferroni(alpha: float, repetitions: int) -> float:
    logging.debug(f"bonferroni <- (alpha={alpha} repetitions={repetitions})")
    return alpha * repetitions


def normal_sample_size(alpha: float, std: float, absolute_error: float) -> int:
    logging.debug(
        f"normal sample size -> (alpha={alpha} std={std} abs_err={absolute_error})"
    )
    return int(math.ceil((__NORMAL__.icdf(alpha / 2) * std / absolute_error) ** 2))


def normal_z(alpha: float) -> float:
    logging.debug(f"normal Z <- (alpha={alpha})")
    return __NORMAL__.icdf(1 - alpha / 2)


def normal_cdf(value: float) -> float:
    logging.debug(f"normal CDF <- (value={value})")
    return __NORMAL__.cdf(value)


def student_sample_size(alpha: float, absolute_error: float) -> float:
    logging.debug(f"student sample size -> (alpha={alpha} abs_err={absolute_error})")
    sample_size = 5
    last_sample_size = -1
    while last_sample_size != sample_size:
        last_sample_size = sample_size
        sample_size = int(
            math.ceil((__student_icdf__(alpha / 2, sample_size) / absolute_error) ** 2)
        )
    return sample_size


def student_z(alpha: float, sample_size: int) -> float:
    logging.debug(f"student Z <- (alpha={alpha} sample_size={sample_size})")
    return __student_icdf__(1 - alpha / 2, sample_size)


def student_cdf(value: float, sample_size: int) -> float:
    logging.debug(f"student cdf -> (value={value} sample_size={sample_size})")
    return __STUDENT__.cdf(value, df=sample_size - 1)


@dataclass
class Measure(ABC):
    name: str
    repeats: int = field(default=1)

    def __hash__(self):
        return hash(self.name)

    @abstractmethod
    def compute_sample_size(
        self,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ) -> int:
        """Compute the sample size to reach the desired error level.
        Relies on the Central Limit Theorem.

        Args:
            error (float): error rate in [0; 1]; interpreted as α (TYPE_I) or β (TYPE_II)
            error_type (ErrorType): whether to control Type I (false positive) or
                Type II (false negative / power) error

        Returns:
            int: sample size required
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_absolute_error(
        self,
        sample_size: int,
        error: float,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ) -> float:
        """Compute absolute error of the measure.
        Relies on the Central Limit Theorem.

        Args:
            sample_size (int): sample size used
            error (float): error rate in [0; 1]; interpreted as α (TYPE_I) or β (TYPE_II)
            error_type (ErrorType): whether to control Type I (false positive) or
                Type II (false negative / power) error

        Returns:
            float: absolute error
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_error_probability(
        self,
        sample_size: int,
        error_type: ErrorType = ErrorType.TYPE_I,
        repetition_multiplier: int = 1,
    ) -> float:
        """Compute the confidence level (TYPE_I) or power (TYPE_II) reached by the target sample size.
        Relies on the Central Limit Theorem.

        Args:
            sample_size (int): sample size used
            error_type (ErrorType): TYPE_I returns confidence level 1 - α;
                TYPE_II returns power 1 - β

        Returns:
            float: [0; 1]
        """
        raise NotImplementedError()

    @abstractmethod
    def test_different(
        self,
        sample1: list[bool],
        sample2: list[bool],
        error: float = 0.05,
        error_type: ErrorType = ErrorType.TYPE_I,
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error (float): error rate for the effect size CI; interpreted as α (TYPE_I)
                or β (TYPE_II)
            error_type (ErrorType): whether to control Type I (false positive) or
                Type II (false negative / power) error

        Returns:
            float: the p-value obtained
            float: effect size
            tuple[float, float]: confidence interval of the effect size
        """
        raise NotImplementedError()
