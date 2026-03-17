from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

__all__ = ["Population", "InfinitePopulation"]


class Population(ABC):
    def is_infinite(self) -> bool:
        """Returns true if this population is infinite."""
        return self.get_size() <= 0

    @abstractmethod
    def get_size(self) -> int:
        """Return the size of this population if this population is infinite returns a negative value."""
        raise NotImplementedError()

    # def filter(self, measure: BooleanMeasure, empirical_proportion: float, error_control: ErrorControl) -> 'FilteredPopulation':
    #     return FilteredPopulation(self, error_control, measure, empirical_proportion)


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


# @dataclass(unsafe_hash=True)
# class FilteredPopulation(Population):
#     source_population: Population
#     error_control: ErrorControl
#     filter_measure: BooleanMeasure
#     empirical_proportion: float

#     def get_size(self):
#         """Produces a conservative estimate of the size of this filtered population."""
#         if self.source_population.is_infinite():
#             return -1
#         else:
#             source_size = self.source_population.get_size()
#             result = int(
#                 math.ceil(
#                     source_size * (self.empirical_proportion + self.filter_measure.absolute_error)
#                 )
#             )
#             logger.debug(
#                 f"conservative estimate of ratio = {self.empirical_proportion + self.filter_measure.absolute_error} to size = {result} from original size= {source_size}"
#             )
#             return result

#     def adjust_error(self, error_control: ErrorControl) -> ErrorControl:
#         return error_control
