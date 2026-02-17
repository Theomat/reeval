from dataclasses import dataclass, field
import logging
from typing import Optional
import math

from reeval.population import FilteredPopulation, FinitePopulation, InfinitePopulation
from reeval.measures import Measure

from reeval.population import Population


logger = logging.getLogger(__name__)

__all__ = ["Evaluation"]


def apply_cochran_finite_pop(pop_size: int, n0: int) -> int:
    logging.debug(f"cochran finite pop. -> (pop_size={pop_size} n0={n0})")
    return int(math.ceil(n0 / (1 + (n0 - 1) / pop_size)))


def reverse_cochran_finite_pop(pop_size: int, n: int) -> int:
    logging.debug(f"cochran finite pop. <- (pop_size={pop_size} n={n})")
    return n * (pop_size - 1) / (pop_size - n)


@dataclass(unsafe_hash=True)
class Evaluation:
    measures: tuple[Measure, ...]
    """The measures present in this evaluation."""
    population: Population = field(default_factory=InfinitePopulation)
    """population the instances are sampled from."""
    repeats: int = field(default=1)
    """The maximum number of times this evaluation will be repeated."""
    confidence: Optional[float] = field(default=None)
    """Confidence level in [0;1] of the evaluation assuming it is the confidence that all statements about all measures are true at the same time and not independently."""

    def __post_init__(self):
        self.measures = tuple(self.measures)

    def _get_total_repeats_(self) -> int:
        return sum(m.repeats for m in self.measures) * self.repeats

    def _raw_sample_size_(self, confidence: float) -> int:
        """Compute the sample size as if pop. was infinite for the given confidence."""
        max_sample_size = 0
        repetition_multiplier = self._get_total_repeats_()
        for measure in self.measures:
            sample_size = measure.compute_sample_size(confidence, repetition_multiplier)
            max_sample_size = max(max_sample_size, sample_size)
        return max_sample_size

    def compute_sample_size(self) -> int:
        """Compute the sample size that ensures all guarantees for all evaluations.

        Returns:
            int: sample size
        """
        logger.debug("computing sample size")

        match self.population:
            case InfinitePopulation():
                return self._raw_sample_size_(self.confidence)
            case FinitePopulation():
                max_sample_size = self._raw_sample_size_(self.confidence)
                logger.debug(
                    "adjusting for finite population size using Cochran's formula"
                )
                return apply_cochran_finite_pop(
                    self.population.get_size(), max_sample_size
                )
            case FilteredPopulation():
                logger.debug("adjusting for filtered population")
                confidence = self.confidence / self.population.filter_confidence
                assert (
                    confidence < 1
                ), "confidence must be lost from a subsequent evaluation!"
                logger.info(
                    f"adjusting confidence from {self.confidence} to {confidence}"
                )

                original_size = self.population.get_size()
                max_sample_size = self._raw_sample_size_(confidence)

                if self.population.is_infinite():
                    return self._raw_sample_size_()
                else:
                    return apply_cochran_finite_pop(max_sample_size, original_size)

    def __get_adjusted_sample_size__(self, sample_size: int) -> int:
        """Inverse sample size corrections to get to the raw uncorrected number."""
        match self.population:
            case InfinitePopulation():
                return sample_size
            case FinitePopulation():
                return reverse_cochran_finite_pop(
                    self.population.get_size(), sample_size
                )
            case FilteredPopulation():
                if self.population.is_infinite():
                    return sample_size
                else:
                    return reverse_cochran_finite_pop(
                        self.population.get_size(), sample_size
                    )

    def compute_confidences(self) -> tuple[float, dict[str, float]]:
        """Compute the total confidence of the evaluation (AND of all statements) and individual (independent) confidence.

        Returns:
            tuple[float, dict[str, float]]: (evaluation confidence, measure name -> confidence)
        """
        logger.debug("computing confidences")

        confs = {}
        total_conf = 1
        sample_size = self.__get_adjusted_sample_size__()
        match self.population:
            case FilteredPopulation():
                total_conf = self.population.filter_confidence

        for measure in self.measures:
            confidence = measure.compute_confidence(sample_size, self.repeats)
            confs[measure.name] = confidence
            total_conf *= confidence

        return total_conf, confs

    def compute_absolute_errors(self) -> dict[str, float]:
        """Compute the absolute error of all measures of this evaluation.

        Returns:
            dict[str, float]: (measure name -> absolute error)
        """
        logger.debug("computing absolute errors")

        total_repeats = self._get_total_repeats_()
        errors = {}
        sample_size = self.__get_adjusted_sample_size__()

        confidence = self.confidence
        match self.population:
            case FilteredPopulation():
                confidence /= self.population.filter_confidence
        repetition_multiplier = total_repeats
        for measure in self.measures:
            error = measure.compute_absolute_error(
                sample_size, confidence, repetition_multiplier
            )
            errors[measure.name] = error

        return errors


def compute_global_sample_sizes(evals: list[Evaluation]) -> dict[Evaluation, int]:
    """Computes the conservative sample size so that all sample size required are respected for all evaluations at the same time.
    Ensures that downstream evaluations reach the desired confidence.

    Args:
        evals (list[Evaluation]): the list of evaluations

    Returns:
        dict[Evaluation, int]: sample size for all the evaluations
    """

    all_evals: dict[Evaluation, int] = {}

    def compute_sample_size_for_eval(eval: Evaluation) -> int:
        sample_size = all_evals.get(eval, eval.compute_sample_size())
        # We need to ensure that in the parent evaluation there is at leas this sample size

        match eval.population:
            case FilteredPopulation():
                if not eval.population.is_infinite():
                    least_ratio = (
                        eval.population.filter_value
                        - eval.population.filter_absolute_error
                    )
                    assert (
                        least_ratio > 0
                    ), "Worst ratio of population with specified property is 0 so sampling cannot be guaranteed, suggestion: decrease the absolute error"
                    new_size = min(
                        int(math.ceil(sample_size / least_ratio)),
                        eval.population.source_population.get_size(),
                    )
                    logger.info(
                        f"going from {sample_size} to {new_size} with least ratio = {least_ratio}"
                    )
                    if new_size > all_evals[eval]:
                        all_evals[eval] = new_size
            case _:
                pass
        all_evals[eval] = sample_size

    while True:
        old = all_evals.copy()
        for ev in evals:
            compute_sample_size_for_eval(ev)
        if all(old[ev] == all_evals[ev] for ev in evals):
            break
    return all_evals
