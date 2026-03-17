from dataclasses import dataclass


__all__ = ["ErrorControl", "AlphaErrorControl", "PowerErrorControl"]


@dataclass(frozen=True)
class ErrorControl:
    @classmethod
    def type_i(cls, alpha: float) -> "AlphaErrorControl":
        return AlphaErrorControl(alpha=alpha)

    @classmethod
    def type_ii(
        cls, beta: float, significance_level: float = 0.05
    ) -> "PowerErrorControl":
        return PowerErrorControl(beta=beta, significance_level=significance_level)


@dataclass(frozen=True)
class AlphaErrorControl(ErrorControl):
    alpha: float

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")


@dataclass(frozen=True)
class PowerErrorControl(ErrorControl):
    beta: float
    significance_level: float = 0.05

    def __post_init__(self):
        if not 0 < self.beta < 1:
            raise ValueError("beta must be strictly between 0 and 1.")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be strictly between 0 and 1.")
