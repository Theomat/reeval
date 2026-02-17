from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import math

from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "Measure",
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
        self, confidence: float, repetition_multiplier: int = 1
    ) -> int:
        """Compute the sample size to reach the desired confidence level.
        Relies on the Central Limit Theorem.

        Args:
            confidence (float): [0; 1]

        Returns:
            int: sample size required
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_absolute_error(
        self, sample_size: int, confidence: float, repetition_multiplier: int = 1
    ) -> float:
        """Compute absolute error of the measure.
        Relies on the Central Limit Theorem.

        Args:
            sample_size (int): sample size used
            confidence (float): [0; 1]

        Returns:
            float: absolute error
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_confidence(
        self, sample_size: int, repetition_multiplier: int = 1
    ) -> float:
        """Compute the confidence level reached by the target sample size.
        Relies on the Central Limit Theorem.

        Args:
            sample_size (int): sample size used

        Returns:
            float: [0; 1]
        """
        raise NotImplementedError()

    @abstractmethod
    def test_different(
        self, sample1: list[bool], sample2: list[bool], confidence: float = 0.95
    ) -> tuple[float, float, tuple[float, float]]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):

        Returns:
            float: the p-value obtained
            float: effect size
            tuple[float, float]: confidence interval of the effect size
        """
        raise NotImplementedError()
