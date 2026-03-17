from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import math

from scipy import stats

from reeval.error_control import ErrorControl

logger = logging.getLogger(__name__)

__all__ = [
    "Measure",
    "ErrorControl",
    "apply_bonferroni",
    "reverse_bonferroni",
    "normal_sample_size",
    "normal_power_sample_size",
    "normal_power",
    "normal_min_detectable_effect",
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


def normal_power(effect_z: float, alpha: float) -> float:
    logging.debug(f"normal power <- (effect_z={effect_z} alpha={alpha})")
    critical_value = normal_z(alpha)
    return normal_cdf(-critical_value - effect_z) + (
        1 - normal_cdf(critical_value - effect_z)
    )


def normal_power_sample_size(
    alpha: float, beta: float, std: float, absolute_error: float
) -> int:
    logging.debug(
        "normal power sample size -> "
        f"(alpha={alpha} beta={beta} std={std} abs_err={absolute_error})"
    )
    z_alpha = normal_z(alpha)
    z_beta = __NORMAL__.icdf(1 - beta)
    return int(math.ceil((((z_alpha + z_beta) * std) / absolute_error) ** 2))


def normal_min_detectable_effect(
    alpha: float, beta: float, std: float, sample_size: int
) -> float:
    logging.debug(
        "normal min detectable effect <- "
        f"(alpha={alpha} beta={beta} std={std} sample_size={sample_size})"
    )
    z_alpha = normal_z(alpha)
    z_beta = __NORMAL__.icdf(1 - beta)
    return ((z_alpha + z_beta) * std) / math.sqrt(sample_size)


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
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ) -> int:
        """Compute the sample size to reach the desired error level.
        Relies on the Central Limit Theorem.

        Args:
            error_control (ErrorControl): complete statistical error specification.

        Returns:
            int: sample size required
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_absolute_error(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ) -> float:
        """Compute absolute error of the measure.
        Relies on the Central Limit Theorem.

        Args:
            sample_size (int): sample size used
            error_control (ErrorControl): complete statistical error specification.

        Returns:
            float: absolute error
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_error_probability(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ) -> float:
        """Compute the confidence level or statistical power reached by the target sample size.
        Relies on the Central Limit Theorem.

        Args:
            sample_size (int): sample size used
            error_control (ErrorControl): alpha control returns confidence level 1 - α;
                power control returns two-sided test power against the configured effect size.

        Returns:
            float: [0; 1]
        """
        raise NotImplementedError()

    @abstractmethod
    def test_different(
        self,
        sample1: list[bool],
        sample2: list[bool],
        error_control: ErrorControl = ErrorControl.type_i(0.05),
    ) -> tuple[float, float, tuple[float, float], float, float]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error_control (ErrorControl): alpha control gives a two-sided
                confidence interval; power control gives the narrower
                beta-calibrated interval.

        Returns:
            float: the p-value obtained
            float: effect size
            tuple[float, float]: confidence interval of the effect size
            float: Type I error (α) for the given sample size
            float: Type II error (β) for the given sample size
        """
        raise NotImplementedError()
