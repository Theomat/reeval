from dataclasses import dataclass, field
import logging
from typing import Optional
import math

from reeval.population import FilteredPopulation, FinitePopulation, InfinitePopulation
from reeval.measure import Measure

from reeval.population import Population


logger = logging.getLogger(__name__)

__all__ = ["Evaluation"]


@dataclass(unsafe_hash=True)
class Evaluation:
    measures: tuple[Measure, ...]
    """The measures present in this evaluation."""
    population: Population = field(default_factory=InfinitePopulation)
    """population the instances are sampled from."""
    max_comparisons: int = field(default=1)
    """The maximum number of times this evaluation will be used to make comparisons, ideally you want that by using an AND of all the comparisons you reach the target confidence level."""
    confidence: Optional[float] = field(default=None)
    """Confidence level in [0;1] of the evaluation assuming it is the confidence that all statements about all measures are true at the same time and not independently."""
    sample_size: Optional[int] = field(default=None)
    """Number of samples taken for this evaluation."""

    def __post_init__(self):
        self.measures = tuple(self.measures)

    def _get_total_repeats_(self) -> int:
        return (
            sum(m._get_adjusted_repetitions_() for m in self.measures)
            * self.max_comparisons
        )

    def _raw_sample_size_(self, confidence: float) -> int:
        """Compute the sample size as if pop. was infinite for the given confidence."""
        max_sample_size = 0
        for measure in self.measures:
            repetition_multiplier = self._get_total_repeats_() / measure.categories
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
                logger.info(
                    "adjusting for finite population size using Cohenn's formula"
                )
                return int(
                    math.ceil(
                        max_sample_size
                        / (1 + (max_sample_size - 1) / self.population.size)
                    )
                )
            case FilteredPopulation():
                logger.debug("adjusting for filtered population")
                confidence = (
                    self.confidence / self.population.filter_evaluation.confidence
                )
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
                    return int(
                        math.ceil(
                            max_sample_size
                            / (1 + (max_sample_size - 1) / original_size)
                        )
                    )

    def __get_adjusted_sample_size__(self) -> int:
        """Inverse sample size corrections to get to the raw uncorrected number."""
        assert self.sample_size is not None, "sample size must be specified"

        match self.population:
            case InfinitePopulation():
                return self.sample_size
            case FinitePopulation():
                logger.info(
                    "adjusting for finite population size using Cohenn's formula"
                )
                return (
                    self.sample_size
                    * (self.population.size - 1)
                    / (self.population.size - self.sample_size)
                )
            case FilteredPopulation():
                raise NotImplementedError()
                logger.info(
                    "adjusting for finite population size using Cohenn's formula"
                )
                return int(
                    math.ceil(
                        self.sample_size
                        / (1 + (self.sample_size - 1) / self.population.size)
                    )
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

        for measure in self.measures:
            before_repeats = measure.repetitions
            measure.repetitions = measure.repetitions * self.max_comparisons
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
        logger.debug("computing absolute errors")

        total_repeats = self._get_total_repeats_()
        errors = {}
        sample_size = self.__get_adjusted_sample_size__()

        for measure in self.measures:
            before_repeats = measure.repetitions
            measure.repetitions = total_repeats / measure.categories
            error = measure.compute_absolute_error(sample_size, self.confidence)
            errors[measure.name] = error
            measure.repetitions = before_repeats

        return errors


def compute_global_sample_sizes(evals: list[Evaluation]) -> dict[Evaluation, int]:
    """Computes the conservative sample size so that all sample size required are respected for all evaluations at the same time.
    Ensures that downstream evaluations reach the desired confidence.

    Args:
        evals (list[Evaluation]): the list of evaluations (the leaves are enough, it needs not to be a tree though merge operators are not yet supported)

    Returns:
        dict[Evaluation, int]: sample size for all the evaluations
    """

    all_evals: dict[Evaluation, int] = {}

    def compute_sample_size_for_eval(eval: Evaluation) -> int:
        sample_size = all_evals.get(eval, eval.compute_sample_size())
        # We need to ensure that in the parent evaluation there is at leas this sample size

        match eval.population:
            case FilteredPopulation():
                if eval.population.is_infinite():
                    raise NotImplementedError()
                else:
                    parent_eval = eval.population.filter_evaluation
                    compute_sample_size_for_eval(parent_eval)
                    least_ratio = 1
                    for m in eval.population.filter_measures:
                        least_ratio *= m.empirical_value - m.absolute_error
                    assert (
                        least_ratio > 0
                    ), "Worst ratio of population with specified property is 0 so sampling cannot be guaranteed, suggestion: decrease the absolute error"
                    new_size = min(
                        int(math.ceil(sample_size / least_ratio)),
                        parent_eval.population.get_size(),
                    )
                    logger.info(
                        f"going from {sample_size} to {new_size} with least ratio = {least_ratio}"
                    )
                    if new_size > all_evals[parent_eval]:
                        all_evals[parent_eval] = new_size
                        compute_sample_size_for_eval(parent_eval)
            case _:
                pass
        all_evals[eval] = sample_size

    for ev in evals:
        compute_sample_size_for_eval(ev)
    return all_evals
