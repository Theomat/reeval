from dataclasses import dataclass, field
import logging
import math

from reeval.error_control import AlphaErrorControl, ErrorControl
from reeval.measures.measure import apply_bonferroni
from reeval.population import FilteredPopulation, InfinitePopulation
from reeval.measures import Measure

from reeval.population import Population


logger = logging.getLogger(__name__)

__all__ = [
    "Evaluation",
    "apply_cochran_finite_pop",
    "reverse_cochran_finite_pop",
    "compute_global_sample_sizes",
    "compute_global_error_probabilities",
    "compute_global_absolute_errors",
]


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
    error_control: ErrorControl = field(
        default_factory=lambda: ErrorControl.type_i(0.05)
    )

    def __post_init__(self):
        self.measures = tuple(self.measures)

    def _get_total_repeats_(self) -> int:
        return sum(m.repeats for m in self.measures)

    def _raw_sample_size_(
        self,
        error_control: ErrorControl,
    ) -> int:
        """Compute the sample size as if pop. was infinite for the given error."""
        max_sample_size = 0
        repetition_multiplier = self._get_total_repeats_()
        if isinstance(error_control, AlphaErrorControl):
            per_measure_error_control = ErrorControl.type_i(
                apply_bonferroni(error_control.alpha, repetition_multiplier)
            )
        else:
            per_measure_error_control = ErrorControl.type_ii(
                apply_bonferroni(error_control.beta, repetition_multiplier),
                significance_level=apply_bonferroni(
                    error_control.significance_level, repetition_multiplier
                ),
            )
        for measure in self.measures:
            sample_size = measure.compute_sample_size(
                per_measure_error_control,
                repetition_multiplier=repetition_multiplier,
            )
            max_sample_size = max(max_sample_size, sample_size)
        return max_sample_size

    def compute_sample_size(self) -> int:
        """Compute the sample size that ensures all guarantees for all evaluations.

        Returns:
            int: sample size
        """
        adjusted_error_control = self.population.adjust_error(self.error_control)
        max_sample_size = self._raw_sample_size_(adjusted_error_control)
        if self.population.is_infinite():
            return max_sample_size

        logger.debug("adjusting for finite population size using Cochran's formula")
        return apply_cochran_finite_pop(self.population.get_size(), max_sample_size)

    def __get_adjusted_sample_size__(self, sample_size: int) -> int:
        """Inverse sample size corrections to get to the raw uncorrected number."""
        if self.population.is_infinite():
            return sample_size
        return reverse_cochran_finite_pop(self.population.get_size(), sample_size)

    def __is_full_population_sample__(self, sample_size: int) -> bool:
        """Return whether the provided sample covers the full finite population."""
        if self.population.is_infinite():
            return False
        return sample_size >= self.population.get_size()

    def compute_error_probability(
        self,
        sample_size: int | None = None,
        error_control: ErrorControl | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute the achieved evaluation-level confidence or power.

        Returns:
            tuple[float, dict[str, float]]: (evaluation confidence/power, measure name -> confidence/power)
        """
        logger.debug("computing confidences")

        confs = {}
        if sample_size is None:
            sample_size = self.compute_sample_size()
        if error_control is None:
            error_control = self.error_control

        if self.__is_full_population_sample__(sample_size):
            for measure in self.measures:
                confs[measure.name] = 1.0

            total_conf = self.population.stage_success_probability()
            for confidence in confs.values():
                total_conf *= confidence
            return total_conf, confs

        sample_size = self.__get_adjusted_sample_size__(sample_size)
        repetition_multiplier = self._get_total_repeats_()

        for measure in self.measures:
            confidence = measure.compute_error_probability(
                sample_size,
                error_control=error_control,
                repetition_multiplier=repetition_multiplier,
            )
            confs[measure.name] = confidence

        total_conf = self.population.stage_success_probability()
        for confidence in confs.values():
            total_conf *= confidence

        return total_conf, confs

    def compute_absolute_errors(
        self,
        sample_size: int | None = None,
        error_control: ErrorControl | None = None,
    ) -> dict[str, float]:
        """Compute the absolute error of all measures of this evaluation.

        Returns:
            dict[str, float]: (measure name -> absolute error)
        """
        logger.debug("computing absolute errors")

        total_repeats = self._get_total_repeats_()
        errors = {}
        if sample_size is None:
            sample_size = self.compute_sample_size()
        if error_control is None:
            error_control = self.error_control

        if self.__is_full_population_sample__(sample_size):
            for measure in self.measures:
                errors[measure.name] = 0.0
            return errors

        sample_size = self.__get_adjusted_sample_size__(sample_size)
        error_control = self.population.adjust_error(error_control)
        repetition_multiplier = total_repeats
        for measure in self.measures:
            abs_error = measure.compute_absolute_error(
                sample_size,
                error_control=error_control,
                repetition_multiplier=repetition_multiplier,
            )
            errors[measure.name] = abs_error

        return errors


def compute_global_sample_sizes(evals: list[Evaluation]) -> dict[Evaluation, int]:
    """Compute mutually compatible sample sizes for chained evaluations.

    Each evaluation first receives its own local sample-size requirement.
    Requirements are then propagated upstream through filtered populations.

    If an evaluation on population `P_f = {x in P : F(x)=1}` requires `n_f`
    filtered examples, and the filtering prevalence has lower conservative
    bound `p_lower`, then any upstream evaluation directly sampling `P` must
    inspect at least `ceil(n_f / p_lower)` items to ensure enough retained
    items for the downstream stage.

    The procedure iterates to a fixed point because filtered populations can be
    chained.
    """

    required_samples = {ev: ev.compute_sample_size() for ev in evals}

    while True:
        previous = required_samples.copy()

        for child_eval in evals:
            child_population = child_eval.population
            if not isinstance(child_population, FilteredPopulation):
                continue

            lower_ratio = child_population.conservative_lower_proportion()
            if lower_ratio <= 0:
                raise ValueError(
                    "Global sample-size propagation is impossible because the "
                    "filtered population may be empty under the conservative "
                    "lower prevalence bound. Increase the empirical proportion "
                    "or decrease the filter absolute error."
                )

            upstream_required = int(
                math.ceil(required_samples[child_eval] / lower_ratio)
            )
            source_population = child_population.source_population
            if (
                not source_population.is_infinite()
                and upstream_required > source_population.get_size()
            ):
                raise ValueError(
                    "Global sample-size propagation is impossible because the "
                    "source population is too small to guarantee enough "
                    "filtered items."
                )

            for parent_eval in evals:
                if parent_eval.population == source_population:
                    required_samples[parent_eval] = max(
                        required_samples[parent_eval], upstream_required
                    )

        if all(previous[ev] == required_samples[ev] for ev in evals):
            return required_samples


def compute_global_error_probabilities(
    evals: list[Evaluation],
    sample_sizes: dict[Evaluation, int],
    error_controls: dict[Evaluation, ErrorControl] | None = None,
) -> dict[Evaluation, tuple[float, dict[str, float]]]:
    """Compute achieved evaluation-level confidence/power for a chained design.

    The caller provides the sample size to use for each evaluation. This keeps
    design (`compute_global_sample_sizes`) separate from reporting.
    """
    results = {}
    for evaluation in evals:
        results[evaluation] = evaluation.compute_error_probability(
            sample_size=sample_sizes[evaluation],
            error_control=None
            if error_controls is None
            else error_controls.get(evaluation, evaluation.error_control),
        )
    return results


def compute_global_absolute_errors(
    evals: list[Evaluation],
    sample_sizes: dict[Evaluation, int],
    error_controls: dict[Evaluation, ErrorControl] | None = None,
) -> dict[Evaluation, dict[str, float]]:
    """Compute achieved absolute errors for a chained design.

    As for `compute_global_error_probabilities`, the caller provides the sample
    size used for each evaluation.
    """
    results = {}
    for evaluation in evals:
        results[evaluation] = evaluation.compute_absolute_errors(
            sample_size=sample_sizes[evaluation],
            error_control=None
            if error_controls is None
            else error_controls.get(evaluation, evaluation.error_control),
        )
    return results
