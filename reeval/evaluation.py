from dataclasses import dataclass, field
import logging
from typing import Optional
import math

from reeval.measure import Measure

logger = logging.getLogger(__name__)

__all__ = ["Evaluation"]


@dataclass
class Evaluation:
    measures: list[Measure]
    confidence: Optional[float] = field(default=None)
    sample_size: Optional[int] = field(default=None)
    population_size: float = field(default=float("inf"))
    max_comparisons: Optional[int] = field(default=None)

    def compute_sample_size(self) -> int:
        """Compute the sample size that ensures all guarantees for all evaluations.

        Returns:
            int: sample size
        """
        logger.info("computing sample size")

        total_repeats = (
            sum(m._get_adjusted_repetitions_() for m in self.measures)
            * self.max_comparisons
        )

        max_sample_size = 0
        for measure in self.measures:
            before_repeats = measure.repetitions
            measure.repetitions = total_repeats / measure.categories
            sample_size = measure.compute_sample_size(self.confidence)
            measure.repetitions = before_repeats
            max_sample_size = max(max_sample_size, sample_size)
        if not math.isinf(self.population_size):
            logger.info("adjusting for finite population size using Cohenn's formula")
            max_sample_size = int(
                math.ceil(
                    max_sample_size / (1 + (max_sample_size - 1) / self.population_size)
                )
            )

        return max_sample_size

    def __get_adjusted_sample_size__(self) -> int:
        assert self.sample_size is not None, "sample size must be specified"
        sample_size = self.sample_size

        if not math.isinf(self.population_size):
            logger.info("adjusting for finite population size using Cohenn's formula")
            sample_size = (
                sample_size
                * (self.population_size - 1)
                / (self.population_size - sample_size)
            )
        return sample_size

    def compute_confidences(self) -> tuple[float, dict[str, float]]:
        """Compute the total confidence of the evaluation (AND of all statements) and individual (independent) confidence.

        Returns:
            tuple[float, dict[str, float]]: (evaluation confidence, measure name -> confidence)
        """
        logger.info("computing confidences")

        total_repeats = (
            sum(m._get_adjusted_repetitions_() for m in self.measures)
            * self.max_comparisons
        )
        confs = {}
        total_conf = 1
        sample_size = self.__get_adjusted_sample_size__()

        for measure in self.measures:
            before_repeats = measure.repetitions
            measure.repetitions = total_repeats / measure.categories
            confidence = measure.compute_confidence(sample_size)
            confs[measure.name] = confidence
            measure.repetitions = before_repeats
            total_conf *= confidence

        return total_conf, confs

    def compute_absolute_errors(self) -> dict[str, float]:
        """Compute the absolute error of all measures of this evaluation.

        Returns:
            dict[str, float]: (measure name -> absolute error)
        """
        logger.info("computing absolute errors")

        total_repeats = (
            sum(m._get_adjusted_repetitions_() for m in self.measures)
            * self.max_comparisons
        )
        errors = {}
        sample_size = self.__get_adjusted_sample_size__()

        for measure in self.measures:
            before_repeats = measure.repetitions
            measure.repetitions = total_repeats / measure.categories
            error = measure.compute_absolute_error(sample_size, self.confidence)
            errors[measure.name] = error
            measure.repetitions = before_repeats

        return errors
