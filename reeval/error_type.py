from enum import Enum


__all__ = ["ErrorType"]


class ErrorType(Enum):
    """The type of statistical error to control in an evaluation."""

    TYPE_I = "type_i"
    """Type I error (false positive rate): the probability of rejecting a true null hypothesis.
    Controlling this limits how often the evaluation incorrectly concludes a difference exists when there is none."""

    TYPE_II = "type_ii"
    """Type II error (false negative rate): the probability of failing to reject a false null hypothesis.
    Controlling this limits how often the evaluation incorrectly concludes no difference exists when one does."""
