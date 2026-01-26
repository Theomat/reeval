from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING

from reeval.measure import MeasureType

if TYPE_CHECKING:
    from reeval.evaluation import Evaluation
    from reeval.measure import Measure

logger = logging.getLogger(__name__)

__all__ = ["Population", "InfinitePopulation", "FilteredPopulation"]


class Population(ABC):
    def filter_on(
        self,
        evaluation: "Evaluation",
        measures: "tuple[Measure, ...] | list[Measure] | Measure",
    ) -> "FilteredPopulation":
        """Filters this population based on the selected measure in the desired evaluation.

        Args:
            evaluation (Evaluation): _description_
            measures (tuple[Measure, ...] | list[Measure] | Measure): boolean or categorical measures

        Returns:
            FilteredPopulation: _description_
        """
        if isinstance(measures, (list, tuple)):
            ms = tuple(measures)
        else:
            ms = (measures,)
        assert all(
            m.measure_type == MeasureType.PROPORTION_BOOLEAN
            or m.measure_type == MeasureType.PROPORTION_CATEGORICAL
            for m in ms
        )
        return FilteredPopulation(self, evaluation, ms)

    def is_infinite(self) -> bool:
        """Returns true if this population is infinite."""
        return self.get_size() <= 0

    @abstractmethod
    def get_size(self) -> int:
        """Return the size of this population if this population is infinite returns a negative value."""
        raise NotImplementedError()


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


@dataclass(unsafe_hash=True)
class FilteredPopulation(Population):
    source_population: Population
    filter_evaluation: "Evaluation"
    filter_measures: tuple["Measure", ...]

    def __post_init__(self):
        self.filter_measures = tuple(self.filter_measures)

    def get_size(self):
        """Produces a conservative estimate of the size of this filtered population."""
        if self.source_population.is_infinite():
            return -1
        else:
            source_size = self.source_population.get_size()
            logger.info(
                f"computing conservative estimate of pop. size from original size= {source_size}"
            )
            filtered_measures = self.filter_measures
            if all(m.empirical_value is not None for m in self.filter_measures):
                worst_case_scenario = 1
                for m in filtered_measures:
                    worst_case_scenario *= m.empirical_value + m.absolute_error
                result = int(math.ceil(source_size * worst_case_scenario))
                logger.info(
                    f"conservative estimate of ratio = {worst_case_scenario} to size = {result}"
                )
                return result
            else:
                raise NotImplementedError()
        return self.size
