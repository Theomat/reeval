from dataclasses import dataclass, field
import math
from reeval.error_control import AlphaErrorControl, ErrorControl
from reeval.measures.measure import (
    Measure,
    apply_bonferroni,
    normal_cdf,
    normal_min_detectable_effect,
    normal_power,
    normal_power_sample_size,
    normal_sample_size,
    normal_z,
    reverse_bonferroni,
    student_cdf,
    student_sample_size,
    student_z,
)
from scipy import stats


__all__ = ["MeanMeasure"]


@dataclass(eq=False)
class MeanMeasure(Measure):
    std: float | None = field(default=None)
    """The standard deviation of the measure at hand.
    """
    absolute_error: float | None = field(default=None)
    """Absolute error of the measure.
    """

    def compute_sample_size(
        self,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            if self.std is None:
                return student_sample_size(alpha, self.absolute_error)
            return normal_sample_size(alpha, self.std, self.absolute_error)
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        beta = apply_bonferroni(
            error_control.beta, self.repeats * repetition_multiplier
        )
        if self.std is None:
            return normal_power_sample_size(alpha, beta, 1.0, self.absolute_error)
        return normal_power_sample_size(alpha, beta, self.std, self.absolute_error)

    def compute_absolute_error(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        if isinstance(error_control, AlphaErrorControl):
            alpha = apply_bonferroni(
                error_control.alpha, self.repeats * repetition_multiplier
            )
            if self.std is None:
                z = student_z(alpha, sample_size)
            else:
                z = normal_z(alpha) * self.std
                return z / math.sqrt(sample_size)
        else:
            alpha = apply_bonferroni(
                error_control.significance_level, self.repeats * repetition_multiplier
            )
            beta = apply_bonferroni(
                error_control.beta, self.repeats * repetition_multiplier
            )
            if self.std is None:
                return normal_min_detectable_effect(alpha, beta, 1.0, sample_size)
            return normal_min_detectable_effect(alpha, beta, self.std, sample_size)
        return z / math.sqrt(sample_size)

    def compute_error_probability(
        self,
        sample_size: int,
        error_control: ErrorControl = ErrorControl.type_i(0.05),
        repetition_multiplier: int = 1,
    ):
        adjusted_sample_size = math.sqrt(sample_size)
        scale = 1.0 if self.std is None else self.std
        if isinstance(error_control, AlphaErrorControl):
            if self.std is None:
                tail_probability = 2 * (
                    1
                    - student_cdf(
                        adjusted_sample_size * self.absolute_error / scale,
                        sample_size,
                    )
                )
            else:
                tail_probability = 2 * (
                    1
                    - normal_cdf(adjusted_sample_size * self.absolute_error / self.std)
                )
            alpha = reverse_bonferroni(
                tail_probability, self.repeats * repetition_multiplier
            )
            alpha = min(max(alpha, 0.0), 1.0)
            return 1 - alpha
        alpha = apply_bonferroni(
            error_control.significance_level, self.repeats * repetition_multiplier
        )
        effect_z = adjusted_sample_size * self.absolute_error / scale
        tail_probability = 1 - normal_power(effect_z, alpha)
        beta = reverse_bonferroni(
            tail_probability, self.repeats * repetition_multiplier
        )
        beta = min(max(beta, 0.0), 1.0)
        return 1 - beta

    def test_different(
        self,
        sample1: list[float],
        sample2: list[float],
        error_control: ErrorControl = ErrorControl.type_i(0.05),
    ) -> tuple[float, float, tuple[float, float], float, float]:
        """Applies a two-tailed test for two samples of the given measure.
        It checks if the parameters are the same.
        It relies on Welch's t-test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error_control (ErrorControl): alpha control gives a two-sided CI;
                power control gives the narrower beta-calibrated interval.

        Returns:
            float: the p-value obtained
            float: effect size (Vargha and Delaney's A12)
            tuple[float, float]: confidence interval of A12
            float: Type I error (α) for the given sample size
            float: Type II error (β) for the given sample size
        """
        result = stats.ttest_ind(sample1, sample2, equal_var=False)
        p_value = result.pvalue
        n = min(len(sample1), len(sample2))
        if isinstance(error_control, AlphaErrorControl):
            z = stats.norm.ppf(1 - error_control.alpha / 2)
            type_i_control = error_control
            type_ii_control = ErrorControl.type_ii(
                error_control.alpha, significance_level=error_control.alpha
            )
        else:
            z = stats.norm.ppf(1 - error_control.beta)
            type_i_control = ErrorControl.type_i(error_control.significance_level)
            type_ii_control = error_control
        type_i_error = 1 - self.compute_error_probability(n, type_i_control)
        type_ii_error = 1 - self.compute_error_probability(n, type_ii_control)
        # When both samples have zero variance (e.g. constant values),
        # ttest_ind returns NaN and mannwhitneyu is unreliable.
        # Treat as no difference: p=1, A12=0.5, degenerate CI.
        if math.isnan(p_value):
            return 1.0, 0.5, (0.5, 0.5), type_i_error, type_ii_error

        n1, n2 = len(sample1), len(sample2)
        # Vargha and Delaney's A12 via Mann-Whitney U
        u_result = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        a12 = u_result.statistic / (n1 * n2)

        # Normal approximation CI for A12
        se = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        ci = (max(0.0, a12 - z * se), min(1.0, a12 + z * se))

        return p_value, a12, ci, type_i_error, type_ii_error

    def test_different_paired_data(
        self,
        sample1: list[float],
        sample2: list[float],
        error_control: ErrorControl = ErrorControl.type_i(0.05),
    ) -> tuple[float, float, tuple[float, float], float, float]:
        """Applies a two-tailed test for two samples where data is paired of the given measure.
        It checks if the parameters are the same.
        It relies on Wilcoxon's test.

        Args:
            sample1 (list[float]):
            sample2 (list[float]):
            error_control (ErrorControl): alpha control gives a two-sided CI;
                power control gives the narrower beta-calibrated interval.

        Returns:
            float: the p-value obtained
            float: effect size (Vargha and Delaney's A12)
            tuple[float, float]: confidence interval of A12
            float: Type I error (α) for the given sample size
            float: Type II error (β) for the given sample size
        """
        result = stats.wilcoxon(sample1, sample2)
        p_value = result.pvalue
        n = min(len(sample1), len(sample2))
        if isinstance(error_control, AlphaErrorControl):
            z = stats.norm.ppf(1 - error_control.alpha / 2)
            type_i_control = error_control
            type_ii_control = ErrorControl.type_ii(
                error_control.alpha, significance_level=error_control.alpha
            )
        else:
            z = stats.norm.ppf(1 - error_control.beta)
            type_i_control = ErrorControl.type_i(error_control.significance_level)
            type_ii_control = error_control
        type_i_error = 1 - self.compute_error_probability(n, type_i_control)
        type_ii_error = 1 - self.compute_error_probability(n, type_ii_control)
        # When both samples have zero variance (e.g. constant values),
        # ttest_ind returns NaN and mannwhitneyu is unreliable.
        # Treat as no difference: p=1, A12=0.5, degenerate CI.
        if math.isnan(p_value):
            return 1.0, 0.5, (0.5, 0.5), type_i_error, type_ii_error

        n1, n2 = len(sample1), len(sample2)
        # Vargha and Delaney's A12 via Mann-Whitney U
        u_result = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        a12 = u_result.statistic / (n1 * n2)

        # Normal approximation CI for A12
        se = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        ci = (max(0.0, a12 - z * se), min(1.0, a12 + z * se))

        return p_value, a12, ci, type_i_error, type_ii_error
