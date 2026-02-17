from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

__all__ = ["Population", "InfinitePopulation", "FilteredPopulation"]


class Population(ABC):
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
    filter_confidence: float
    filter_value: float
    filter_absolute_error: float

    def get_size(self):
        """Produces a conservative estimate of the size of this filtered population."""
        if self.source_population.is_infinite():
            return -1
        else:
            source_size = self.source_population.get_size()
            logger.info(
                f"computing conservative estimate of pop. size from original size= {source_size}"
            )
            result = int(
                math.ceil(
                    source_size * (self.filter_value + self.filter_absolute_error)
                )
            )
            logger.info(
                f"conservative estimate of ratio = {self.filter_value + self.filter_absolute_error} to size = {result}"
            )
            return result
