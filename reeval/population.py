from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math

from reeval.error_control import AlphaErrorControl, ErrorControl
from reeval.measures.boolean_measure import BooleanMeasure


logger = logging.getLogger(__name__)

__all__ = [
    "Population",
    "FinitePopulation",
    "InfinitePopulation",
    "FilteredPopulation",
]


class Population(ABC):
    def is_infinite(self) -> bool:
        """Returns true if this population is infinite."""
        return self.get_size() <= 0

    @abstractmethod
    def get_size(self) -> int:
        """Return the size of this population if this population is infinite returns a negative value."""
        raise NotImplementedError()

    def filter(
        self,
        measure: BooleanMeasure,
        empirical_proportion: float,
        error_control: ErrorControl,
    ) -> "FilteredPopulation":
        """Create the downstream population induced by a boolean filter.

        The empirical proportion is the observed prevalence of the filtering
        predicate in the source population sample. The filter measure's
        absolute error is then used to conservatively upper bound the size of
        the retained finite population.
        """

        return FilteredPopulation(self, error_control, measure, empirical_proportion)

    def stage_success_probability(self) -> float:
        """Success probability contributed by the population-construction stage."""

        return 1.0

    def adjust_error(self, error_control: ErrorControl) -> ErrorControl:
        """Tighten downstream error control to preserve the requested joint guarantee."""

        return error_control


@dataclass(frozen=True)
class FinitePopulation(Population):
    size: int

    def get_size(self):
        return self.size


class InfinitePopulation(Population):
    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, InfinitePopulation)

    def get_size(self):
        return -1


@dataclass(frozen=True)
class FilteredPopulation(Population):
    """Population obtained by retaining only items satisfying a boolean predicate.

    Mathematically, if the observed prevalence of the predicate is `p_hat` and
    the corresponding boolean measure has absolute error `eps`, then the true
    prevalence is upper bounded by `p_hat + eps` with the filter evaluation's
    guarantee probability. For finite source populations we therefore use the
    conservative cardinality bound `ceil(N * min(1, p_hat + eps))`.

    Any downstream evaluation must also account for the probability mass
    already consumed by the filtering step. Assuming independence is not
    required: by the product rule, guaranteeing a joint success probability
    `g_target` after a prior step with guarantee `g_filter` requires the
    downstream step to satisfy at least `g_target / g_filter`.
    """

    source_population: Population
    error_control: ErrorControl
    filter_measure: BooleanMeasure
    empirical_proportion: float

    def __post_init__(self):
        if not 0 <= self.empirical_proportion <= 1:
            raise ValueError("empirical_proportion must lie in [0, 1].")
        if self.filter_measure.absolute_error is None:
            raise ValueError(
                "filter_measure.absolute_error must be defined for a filtered population."
            )

    def get_size(self) -> int:
        """Produce a conservative upper bound on the filtered population size."""

        if self.source_population.is_infinite():
            return -1

        source_size = self.source_population.get_size()
        conservative_ratio = self.conservative_upper_proportion()
        result = int(math.ceil(source_size * conservative_ratio))
        logger.debug(
            "conservative filtered population size: ratio=%s size=%s source_size=%s",
            conservative_ratio,
            result,
            source_size,
        )
        return result

    def conservative_upper_proportion(self) -> float:
        """Upper confidence/power bound on the filtered prevalence."""

        return min(1.0, self.empirical_proportion + self.filter_measure.absolute_error)

    def conservative_lower_proportion(self) -> float:
        """Lower confidence/power bound on the filtered prevalence."""

        return max(0.0, self.empirical_proportion - self.filter_measure.absolute_error)

    def stage_success_probability(self) -> float:
        if isinstance(self.error_control, AlphaErrorControl):
            return 1 - self.error_control.alpha
        return 1 - self.error_control.beta

    def adjust_error(self, error_control: ErrorControl) -> ErrorControl:
        """Tighten downstream error control to preserve the requested joint guarantee."""

        filter_guarantee = self.stage_success_probability()
        if filter_guarantee >= 1:
            return error_control

        if isinstance(error_control, AlphaErrorControl):
            target_guarantee = 1 - error_control.alpha
            adjusted_guarantee = target_guarantee / filter_guarantee
            if adjusted_guarantee >= 1:
                raise ValueError(
                    "Requested Type I guarantee is unattainable after filtering: "
                    f"target={target_guarantee}, filter_guarantee={filter_guarantee}."
                )
            return ErrorControl.type_i(1 - adjusted_guarantee)

        target_guarantee = 1 - error_control.beta
        adjusted_guarantee = target_guarantee / filter_guarantee
        if adjusted_guarantee >= 1:
            raise ValueError(
                "Requested power is unattainable after filtering: "
                f"target={target_guarantee}, filter_guarantee={filter_guarantee}."
            )
        return ErrorControl.type_ii(
            1 - adjusted_guarantee,
            significance_level=error_control.significance_level,
        )
