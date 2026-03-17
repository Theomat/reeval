import math
import random

import pytest

from reeval.error_control import ErrorControl

from reeval.measures.boolean_measure import BooleanMeasure, CategoricalMeasures
from reeval.measures.mean_measure import MeanMeasure
from reeval.measures.variance_measure import VarianceMeasure
from reeval.measures.rank_measure import RankMeasure


POWER_ALPHA = 0.05


def ec_i(alpha: float = 0.05):
    return ErrorControl.type_i(alpha)


def ec_ii(beta: float = 0.05, alpha: float = POWER_ALPHA):
    return ErrorControl.type_ii(beta, significance_level=alpha)


# ---------------------------------------------------------------------------
# Fixtures – one representative instance per measure type
# ---------------------------------------------------------------------------


@pytest.fixture
def boolean_measure():
    return BooleanMeasure(name="bool", absolute_error=0.05)


@pytest.fixture
def boolean_measure_with_std():
    return BooleanMeasure(name="bool_std", std=0.4, absolute_error=0.05)


@pytest.fixture
def mean_measure_known_std():
    return MeanMeasure(name="mean_known", std=1.0, absolute_error=0.1)


@pytest.fixture
def variance_measure():
    return VarianceMeasure(name="var", relative_error=0.1)


# All measures whose compute_sample_size is known to work.
# mean_unknown is excluded (student_sample_size infinite loop)
# rank is excluded (self.k AttributeError – should be self.max_rank)
@pytest.fixture(params=["boolean", "boolean_std", "mean_known", "variance"])
def any_measure(request):
    factories = {
        "boolean": lambda: BooleanMeasure(name="bool", absolute_error=0.05),
        "boolean_std": lambda: BooleanMeasure(
            name="bool_std", std=0.4, absolute_error=0.05
        ),
        "mean_known": lambda: MeanMeasure(
            name="mean_known", std=1.0, absolute_error=0.1
        ),
        "variance": lambda: VarianceMeasure(name="var", relative_error=0.1),
    }
    return factories[request.param]()


# Measures that support test_different
@pytest.fixture(params=["boolean", "mean_known"])
def testable_measure(request):
    factories = {
        "boolean": lambda: BooleanMeasure(name="bool", absolute_error=0.05),
        "mean_known": lambda: MeanMeasure(
            name="mean_known", std=1.0, absolute_error=0.1
        ),
    }
    return factories[request.param]()


# =========================================================================
# 1. compute_sample_size – general properties
# =========================================================================


class TestComputeSampleSize:
    def test_returns_positive_integer(self, any_measure):
        n = any_measure.compute_sample_size(error_control=ec_i(0.05))
        assert isinstance(n, (int, float))
        assert n > 0

    def test_monotone_increasing_with_confidence(self, any_measure):
        """Lower error (higher confidence) must require at least as many samples."""
        n_low = any_measure.compute_sample_size(error_control=ec_i(0.20))
        n_high = any_measure.compute_sample_size(error_control=ec_i(0.01))
        assert n_high >= n_low

    def test_increases_with_repetition_multiplier(self, any_measure):
        """More simultaneous comparisons (Bonferroni) should not decrease the
        required sample size."""
        n1 = any_measure.compute_sample_size(
            error_control=ec_i(0.05), repetition_multiplier=1
        )
        n2 = any_measure.compute_sample_size(
            error_control=ec_i(0.05), repetition_multiplier=5
        )
        assert n2 >= n1

    def test_increases_with_repeats(self):
        """More built-in repeats (Bonferroni) should not decrease sample size."""
        m1 = BooleanMeasure(name="b", repeats=1, absolute_error=0.05)
        m2 = BooleanMeasure(name="b", repeats=10, absolute_error=0.05)
        assert m2.compute_sample_size(ec_i(0.05)) >= m1.compute_sample_size(ec_i(0.05))


class TestSampleSizeDecreasesWithTolerance:
    """Larger error tolerance should require fewer samples."""

    def test_boolean_measure(self):
        m_tight = BooleanMeasure(name="b", absolute_error=0.01)
        m_loose = BooleanMeasure(name="b", absolute_error=0.10)
        assert m_tight.compute_sample_size(ec_i(0.05)) > m_loose.compute_sample_size(
            ec_i(0.05)
        )

    def test_mean_measure_known_std(self):
        m_tight = MeanMeasure(name="m", std=1.0, absolute_error=0.01)
        m_loose = MeanMeasure(name="m", std=1.0, absolute_error=0.10)
        assert m_tight.compute_sample_size(ec_i(0.05)) > m_loose.compute_sample_size(
            ec_i(0.05)
        )

    def test_variance_measure(self):
        m_tight = VarianceMeasure(name="v", relative_error=0.01)
        m_loose = VarianceMeasure(name="v", relative_error=0.10)
        assert m_tight.compute_sample_size(ec_i(0.05)) > m_loose.compute_sample_size(
            ec_i(0.05)
        )

    def test_rank_measure(self):
        m_tight = RankMeasure(name="r", max_rank=10, absolute_error=0.1)
        m_loose = RankMeasure(name="r", max_rank=10, absolute_error=1.0)
        assert m_tight.compute_sample_size(ec_i(0.05)) > m_loose.compute_sample_size(
            ec_i(0.05)
        )

    def test_mean_measure_unknown_std(self):
        m_tight = MeanMeasure(name="m", absolute_error=0.01)
        m_loose = MeanMeasure(name="m", absolute_error=0.10)
        assert m_tight.compute_sample_size(ec_i(0.05)) > m_loose.compute_sample_size(
            ec_i(0.05)
        )


class TestSampleSizeIncreasesWithUncertainty:
    """More uncertain distributions need more samples."""

    def test_boolean_higher_std_needs_more_samples(self):
        m_low = BooleanMeasure(name="b", std=0.1, absolute_error=0.05)
        m_high = BooleanMeasure(name="b", std=0.5, absolute_error=0.05)
        assert m_high.compute_sample_size(ec_i(0.05)) >= m_low.compute_sample_size(
            ec_i(0.05)
        )

    def test_mean_higher_std_needs_more_samples(self):
        m_low = MeanMeasure(name="m", std=0.5, absolute_error=0.1)
        m_high = MeanMeasure(name="m", std=2.0, absolute_error=0.1)
        assert m_high.compute_sample_size(ec_i(0.05)) >= m_low.compute_sample_size(
            ec_i(0.05)
        )

    def test_rank_higher_max_rank_needs_more_samples(self):
        m_small = RankMeasure(name="r", max_rank=3, absolute_error=0.5)
        m_large = RankMeasure(name="r", max_rank=20, absolute_error=0.5)
        assert m_large.compute_sample_size(ec_i(0.05)) >= m_small.compute_sample_size(
            ec_i(0.05)
        )

    @pytest.mark.parametrize("sensitivity", [0.99, 0.95, 0.9, 0.75])
    def test_boolean_lower_sensitivity_needs_more_samples(self, sensitivity):
        baseline = BooleanMeasure(
            name="b", absolute_error=0.05, sensitivity=1.0, specificity=1.0
        )
        adjusted = BooleanMeasure(
            name="b",
            absolute_error=0.05,
            sensitivity=sensitivity,
            specificity=1.0,
        )

        assert adjusted.compute_sample_size(ec_i(0.05)) >= baseline.compute_sample_size(
            ec_i(0.05)
        )

    @pytest.mark.parametrize("specificity", [0.99, 0.95, 0.9, 0.75])
    def test_boolean_lower_specificity_needs_more_samples(self, specificity):
        baseline = BooleanMeasure(
            name="b", absolute_error=0.05, sensitivity=1.0, specificity=1.0
        )
        adjusted = BooleanMeasure(
            name="b",
            absolute_error=0.05,
            sensitivity=1.0,
            specificity=specificity,
        )

        assert adjusted.compute_sample_size(ec_i(0.05)) >= baseline.compute_sample_size(
            ec_i(0.05)
        )

    @pytest.mark.parametrize("quality", [0.99, 0.95, 0.9, 0.75])
    def test_boolean_lower_joint_label_quality_needs_more_samples(self, quality):
        baseline = BooleanMeasure(
            name="b", absolute_error=0.05, sensitivity=1.0, specificity=1.0
        )
        adjusted = BooleanMeasure(
            name="b",
            absolute_error=0.05,
            sensitivity=quality,
            specificity=quality,
        )

        assert adjusted.compute_sample_size(ec_i(0.05)) >= baseline.compute_sample_size(
            ec_i(0.05)
        )


# =========================================================================
# 1b. compute_sample_size – TYPE_II properties
# =========================================================================


class TestComputeSampleSizeTypeII:
    """Type II error control (power analysis) properties."""

    def test_returns_positive(self):
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(
            error_control=ec_ii(0.20),
        )
        assert n > 0

    def test_type_ii_more_samples_than_type_i(self):
        """At the same nominal error level, classical power analysis adds both
        z_{α/2} and z_{1-β}, so TYPE_II needs at least as many samples as TYPE_I."""
        for m in [
            BooleanMeasure(name="b", absolute_error=0.05),
            MeanMeasure(name="m", std=1.0, absolute_error=0.1),
            RankMeasure(name="r", max_rank=10, absolute_error=0.5),
            VarianceMeasure(name="v", relative_error=0.1),
        ]:
            n_i = m.compute_sample_size(ec_i(0.05))
            n_ii = m.compute_sample_size(ec_ii(0.05))
            assert n_ii >= n_i, f"{type(m).__name__}: expected n_II >= n_I"

    def test_monotone_increasing_with_power(self):
        """Higher power (lower β) requires more samples."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n_low_power = m.compute_sample_size(
            error_control=ec_ii(0.20),
        )
        n_high_power = m.compute_sample_size(
            error_control=ec_ii(0.05),
        )
        assert n_high_power >= n_low_power

    def test_increases_with_repetition_multiplier(self):
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n1 = m.compute_sample_size(
            error_control=ec_ii(0.20),
            repetition_multiplier=1,
        )
        n5 = m.compute_sample_size(
            error_control=ec_ii(0.20),
            repetition_multiplier=5,
        )
        assert n5 >= n1


# =========================================================================
# 2. compute_error_probability – general properties
# =========================================================================


class TestComputeConfidence:
    def test_returns_value_in_valid_range(self, any_measure):
        """Confidence should be in (0, 1] for reasonable sample sizes."""
        n = any_measure.compute_sample_size(error_control=ec_i(0.05))
        conf = any_measure.compute_error_probability(n, ec_i())
        assert 0 < conf <= 1.0

    def test_monotone_increasing_with_sample_size(self, any_measure):
        """More samples should not decrease confidence."""
        n_base = any_measure.compute_sample_size(error_control=ec_i(0.10))
        conf_small = any_measure.compute_error_probability(
            max(n_base, 10), error_control=ec_i()
        )
        conf_large = any_measure.compute_error_probability(
            max(n_base * 5, 50), error_control=ec_i()
        )
        assert conf_large >= conf_small

    def test_more_repeats_reduce_confidence(self):
        """Bonferroni correction: more repeats at same sample size -> lower confidence."""
        m1 = BooleanMeasure(name="b", repeats=1, absolute_error=0.05)
        m2 = BooleanMeasure(name="b", repeats=10, absolute_error=0.05)
        n = 500
        assert m1.compute_error_probability(n, ec_i()) >= m2.compute_error_probability(
            n, ec_i()
        )


# =========================================================================
# 2b. compute_error_probability – TYPE_II (power) properties
# =========================================================================


class TestComputeConfidenceTypeII:
    """Type II error control: compute_error_probability returns power (1 - β)."""

    def test_returns_value_in_valid_range(self, any_measure):
        """Power should be in (0, 1] for reasonable sample sizes."""
        n = any_measure.compute_sample_size(
            error_control=ec_ii(0.20),
        )
        power = any_measure.compute_error_probability(
            n,
            error_control=ec_ii(),
        )
        assert 0 < power <= 1.0

    def test_monotone_increasing_with_sample_size(self, any_measure):
        """More samples should not decrease power."""
        n_base = any_measure.compute_sample_size(
            error_control=ec_ii(0.20),
        )
        power_small = any_measure.compute_error_probability(
            max(n_base, 10),
            error_control=ec_ii(),
        )
        power_large = any_measure.compute_error_probability(
            max(n_base * 5, 50),
            error_control=ec_ii(),
        )
        assert power_large >= power_small

    def test_more_repeats_reduce_power(self):
        """Bonferroni correction: more repeats at same sample size -> lower power."""
        m1 = BooleanMeasure(name="b", repeats=1, absolute_error=0.05)
        m2 = BooleanMeasure(name="b", repeats=10, absolute_error=0.05)
        n = 500
        assert m1.compute_error_probability(n, ec_ii()) >= m2.compute_error_probability(
            n, ec_ii()
        )

    def test_type_ii_lower_than_type_i_near_type_i_threshold(self):
        """At a sample size calibrated only for TYPE_I confidence, actual test power is
        typically lower because the rejection threshold α must also be crossed."""
        for m in [
            BooleanMeasure(name="b", absolute_error=0.05),
            MeanMeasure(name="m", std=1.0, absolute_error=0.1),
            RankMeasure(name="r", max_rank=10, absolute_error=0.5),
            VarianceMeasure(name="v", relative_error=0.1),
        ]:
            n = m.compute_sample_size(ec_i(0.05))
            conf_i = m.compute_error_probability(n, ec_i())
            power_ii = m.compute_error_probability(
                n,
                error_control=ec_ii(),
            )
            assert power_ii <= conf_i
            assert conf_i - power_ii > 1e-6, (
                f"{type(m).__name__}: expected TYPE_I confidence to exceed "
                f"TYPE_II power at the TYPE_I-calibrated sample size ({conf_i} vs {power_ii})"
            )


# =========================================================================
# 3. compute_absolute_error – general properties (not on VarianceMeasure)
# =========================================================================


@pytest.fixture(params=["boolean", "boolean_std", "mean_known"])
def error_measure(request):
    factories = {
        "boolean": lambda: BooleanMeasure(name="bool", absolute_error=0.05),
        "boolean_std": lambda: BooleanMeasure(
            name="bool_std", std=0.4, absolute_error=0.05
        ),
        "mean_known": lambda: MeanMeasure(
            name="mean_known", std=1.0, absolute_error=0.1
        ),
    }
    return factories[request.param]()


class TestComputeAbsoluteError:
    def test_returns_positive(self, error_measure):
        err = error_measure.compute_absolute_error(
            sample_size=100, error_control=ec_i(0.05)
        )
        assert err > 0

    def test_decreases_with_sample_size(self, error_measure):
        err_small = error_measure.compute_absolute_error(
            sample_size=50, error_control=ec_i(0.05)
        )
        err_large = error_measure.compute_absolute_error(
            sample_size=500, error_control=ec_i(0.05)
        )
        assert err_large < err_small

    def test_increases_with_confidence(self, error_measure):
        """At the same sample size, lower error (higher confidence) -> wider error bar."""
        err_low = error_measure.compute_absolute_error(
            sample_size=200, error_control=ec_i(0.20)
        )
        err_high = error_measure.compute_absolute_error(
            sample_size=200, error_control=ec_i(0.01)
        )
        assert err_high >= err_low

    def test_variance_measure_raises(self, variance_measure):
        with pytest.raises(NotImplementedError):
            variance_measure.compute_absolute_error(
                sample_size=100, error_control=ec_i(0.05)
            )

    def test_rank_absolute_error_decreases_with_sample_size(self):
        m = RankMeasure(name="rank", max_rank=10, absolute_error=0.5)
        err_small = m.compute_absolute_error(sample_size=50, error_control=ec_i(0.05))
        err_large = m.compute_absolute_error(sample_size=500, error_control=ec_i(0.05))
        assert err_large < err_small


# =========================================================================
# 3b. compute_absolute_error – TYPE_II properties
# =========================================================================


class TestComputeAbsoluteErrorTypeII:
    """Type II error control properties for absolute error."""

    def test_returns_positive(self):
        m = BooleanMeasure(name="b", absolute_error=0.05)
        err = m.compute_absolute_error(
            sample_size=100,
            error_control=ec_ii(0.20),
        )
        assert err > 0

    def test_type_ii_larger_than_type_i(self):
        """Classical power requires clearing the significance threshold and the desired
        miss-rate target, so the minimum detectable effect is wider than TYPE_I."""
        for m in [
            BooleanMeasure(name="b", absolute_error=0.05),
            RankMeasure(name="r", max_rank=10, absolute_error=0.5),
        ]:
            e_i = m.compute_absolute_error(sample_size=100, error_control=ec_i(0.05))
            e_ii = m.compute_absolute_error(
                sample_size=100,
                error_control=ec_ii(0.05),
            )
            assert e_ii >= e_i, f"{type(m).__name__}: expected TYPE_II error >= TYPE_I"

    def test_decreases_with_sample_size(self):
        m = BooleanMeasure(name="b", absolute_error=0.05)
        err_small = m.compute_absolute_error(
            sample_size=50,
            error_control=ec_ii(0.20),
        )
        err_large = m.compute_absolute_error(
            sample_size=500,
            error_control=ec_ii(0.20),
        )
        assert err_large < err_small

    def test_increases_with_power(self):
        """Lower β (higher power) -> larger z_β -> wider detectable-effect bound."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        err_low_power = m.compute_absolute_error(
            sample_size=200,
            error_control=ec_ii(0.20),
        )
        err_high_power = m.compute_absolute_error(
            sample_size=200,
            error_control=ec_ii(0.01),
        )
        assert err_high_power >= err_low_power


# =========================================================================
# 4. VarianceMeasure.compute_relative_error – general properties
# =========================================================================


class TestVarianceRelativeError:
    def test_returns_positive(self, variance_measure):
        err = variance_measure.compute_relative_error(
            sample_size=100, error_control=ec_i(0.05)
        )
        assert err > 0

    def test_decreases_with_sample_size(self, variance_measure):
        err_small = variance_measure.compute_relative_error(
            sample_size=50, error_control=ec_i(0.05)
        )
        err_large = variance_measure.compute_relative_error(
            sample_size=500, error_control=ec_i(0.05)
        )
        assert abs(err_large) < abs(err_small)

    def test_increases_with_confidence(self, variance_measure):
        """At the same sample size, lower error (higher confidence) -> wider relative error."""
        err_low = variance_measure.compute_relative_error(
            sample_size=200, error_control=ec_i(0.20)
        )
        err_high = variance_measure.compute_relative_error(
            sample_size=200, error_control=ec_i(0.01)
        )
        assert abs(err_high) >= abs(err_low)

    def test_type_ii_larger_than_type_i(self, variance_measure):
        """Classical power requires a larger detectable relative effect than TYPE_I."""
        err_i = variance_measure.compute_relative_error(
            sample_size=200, error_control=ec_i(0.05)
        )
        err_ii = variance_measure.compute_relative_error(
            sample_size=200,
            error_control=ec_ii(0.05),
        )
        assert abs(err_ii) >= abs(err_i)


# =========================================================================
# 5. test_different – general properties
# =========================================================================


class TestTestDifferent:
    def test_p_value_in_range(self, testable_measure):
        """p-value must be in [0, 1]."""
        random.seed(42)
        if isinstance(testable_measure, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(50)]
            s2 = [random.choice([True, False]) for _ in range(50)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(50)]
            s2 = [random.gauss(0, 1) for _ in range(50)]
        p, effect, ci, type_i_error, type_ii_error = testable_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert 0 <= p <= 1

    def test_identical_samples_high_p_value(self, testable_measure):
        """Two identical samples should not be flagged as different."""
        random.seed(123)
        if isinstance(testable_measure, BooleanMeasure):
            s = [random.choice([True, False]) for _ in range(100)]
        else:
            s = [random.gauss(5, 1) for _ in range(100)]
        p, *_ = testable_measure.test_different(s, s, error_control=ec_i(0.05))
        assert (
            p >= 0.05
        ), f"Identical samples should not be significantly different, got p={p}"

    def test_very_different_samples_low_p_value(self, testable_measure):
        """Two clearly different distributions should be detected."""
        n = 200
        random.seed(999)
        if isinstance(testable_measure, BooleanMeasure):
            s1 = [True] * n
            s2 = [False] * n
        else:
            s1 = [random.gauss(0, 0.1) for _ in range(n)]
            s2 = [random.gauss(100, 0.1) for _ in range(n)]
        p, *_ = testable_measure.test_different(s1, s2, error_control=ec_i(0.05))
        assert p < 0.01, f"Very different samples should be detected, got p={p}"

    def test_returns_three_values(self, testable_measure):
        """test_different should return (p_value, effect_size, ci_tuple)."""
        random.seed(0)
        if isinstance(testable_measure, BooleanMeasure):
            s1 = [True, False, True, True, False] * 10
            s2 = [False, True, False, False, True] * 10
        else:
            s1 = [float(i) for i in range(50)]
            s2 = [float(i) + 10 for i in range(50)]
        result = testable_measure.test_different(s1, s2, error_control=ec_i(0.05))
        assert len(result) == 5
        p_value, effect_size, ci, type_i_error, type_ii_error = result
        assert isinstance(p_value, float)
        assert isinstance(effect_size, float)
        assert isinstance(ci, tuple) and len(ci) == 2
        assert isinstance(type_i_error, float)
        assert isinstance(type_ii_error, float)

    def test_ci_lower_le_upper(self, testable_measure):
        """Confidence interval lower bound should be <= upper bound."""
        random.seed(7)
        if isinstance(testable_measure, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(60)]
            s2 = [random.choice([True, False]) for _ in range(60)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(60)]
            s2 = [random.gauss(0, 1) for _ in range(60)]
        _, _, (lo, hi), *_ = testable_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert lo <= hi

    def test_symmetry_of_p_value(self, testable_measure):
        """Swapping sample1 and sample2 should not change the p-value."""
        random.seed(55)
        if isinstance(testable_measure, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(40)]
            s2 = [random.choice([True, False]) for _ in range(40)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(40)]
            s2 = [random.gauss(1, 1) for _ in range(40)]
        p_forward, *_ = testable_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        p_reverse, *_ = testable_measure.test_different(
            s2, s1, error_control=ec_i(0.05)
        )
        assert abs(p_forward - p_reverse) < 1e-10

    def test_rank_test_different_p_in_range(self):
        m = RankMeasure(name="rank", max_rank=10, absolute_error=0.5)
        random.seed(42)
        s1 = [random.randint(1, 10) for _ in range(50)]
        s2 = [random.randint(1, 10) for _ in range(50)]
        p, *_ = m.test_different(s1, s2, error_control=ec_i(0.05))
        assert 0 <= p <= 1


# =========================================================================
# 5b. Effect size properties
# =========================================================================


class TestEffectSizeA12:
    """Properties of Vargha-Delaney A12 (MeanMeasure and RankMeasure)."""

    @pytest.fixture(params=["mean", "rank"])
    def a12_measure(self, request):
        factories = {
            "mean": lambda: MeanMeasure(name="m", std=1.0, absolute_error=0.1),
            "rank": lambda: RankMeasure(name="r", max_rank=10, absolute_error=0.5),
        }
        return factories[request.param]()

    def test_a12_bounded(self, a12_measure):
        """A12 must lie in [0, 1]."""
        random.seed(42)
        s1 = [random.gauss(0, 1) for _ in range(50)]
        s2 = [random.gauss(1, 1) for _ in range(50)]
        _, a12, *_ = a12_measure.test_different(s1, s2, error_control=ec_i(0.05))
        assert 0 <= a12 <= 1

    def test_a12_no_effect_for_identical_samples(self, a12_measure):
        """Identical distributions should yield A12 ≈ 0.5."""
        random.seed(10)
        s1 = [random.gauss(5, 1) for _ in range(200)]
        s2 = [random.gauss(5, 1) for _ in range(200)]
        _, a12, *_ = a12_measure.test_different(s1, s2, error_control=ec_i(0.05))
        assert (
            abs(a12 - 0.5) < 0.1
        ), f"Expected A12 ≈ 0.5 for similar samples, got {a12}"

    def test_a12_large_effect_for_separated_samples(self, a12_measure):
        """Clearly separated samples should yield A12 near 0 or 1."""
        random.seed(77)
        s1 = [random.gauss(0, 0.1) for _ in range(100)]
        s2 = [random.gauss(100, 0.1) for _ in range(100)]
        _, a12, *_ = a12_measure.test_different(s1, s2, error_control=ec_i(0.05))
        assert a12 > 0.9 or a12 < 0.1, f"Expected extreme A12, got {a12}"

    def test_a12_antisymmetry(self, a12_measure):
        """Swapping samples: A12_forward + A12_reverse ≈ 1."""
        random.seed(33)
        s1 = [random.gauss(0, 1) for _ in range(60)]
        s2 = [random.gauss(2, 1) for _ in range(60)]
        _, a12_fwd, *_ = a12_measure.test_different(s1, s2, error_control=ec_i(0.05))
        _, a12_rev, *_ = a12_measure.test_different(s2, s1, error_control=ec_i(0.05))
        assert (
            abs((a12_fwd + a12_rev) - 1.0) < 1e-10
        ), f"A12 antisymmetry violated: {a12_fwd} + {a12_rev} != 1"

    def test_a12_ci_contains_point_estimate(self, a12_measure):
        """The confidence interval should contain the A12 point estimate."""
        random.seed(21)
        s1 = [random.gauss(0, 1) for _ in range(80)]
        s2 = [random.gauss(1, 1) for _ in range(80)]
        _, a12, (lo, hi), *_ = a12_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert lo <= a12 <= hi, f"CI [{lo}, {hi}] does not contain A12={a12}"

    def test_a12_ci_bounded(self, a12_measure):
        """A12 confidence interval must be within [0, 1]."""
        random.seed(99)
        s1 = [random.gauss(0, 1) for _ in range(50)]
        s2 = [random.gauss(0, 1) for _ in range(50)]
        _, _, (lo, hi), *_ = a12_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert 0 <= lo <= hi <= 1

    def test_a12_ci_narrows_with_more_samples(self, a12_measure):
        """Larger samples should produce a tighter CI."""
        random.seed(55)
        s1_small = [random.gauss(0, 1) for _ in range(30)]
        s2_small = [random.gauss(1, 1) for _ in range(30)]
        _, _, (lo_s, hi_s), *_ = a12_measure.test_different(
            s1_small, s2_small, error_control=ec_i(0.05)
        )

        random.seed(55)
        s1_large = [random.gauss(0, 1) for _ in range(300)]
        s2_large = [random.gauss(1, 1) for _ in range(300)]
        _, _, (lo_l, hi_l), *_ = a12_measure.test_different(
            s1_large, s2_large, error_control=ec_i(0.05)
        )

        assert (hi_l - lo_l) < (hi_s - lo_s), "CI should narrow with more samples"


class TestEffectSizeOddsRatio:
    """Properties of the odds ratio effect size (BooleanMeasure)."""

    @pytest.fixture
    def bool_measure(self):
        return BooleanMeasure(name="b", absolute_error=0.05)

    def test_or_non_negative(self, bool_measure):
        """Odds ratio must be >= 0."""
        random.seed(42)
        s1 = [random.choice([True, False]) for _ in range(80)]
        s2 = [random.choice([True, False]) for _ in range(80)]
        _, odds_ratio, *_ = bool_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert odds_ratio >= 0

    def test_or_no_effect_for_identical_samples(self, bool_measure):
        """Identical samples should yield OR = 1."""
        s = [True, False, True, True, False] * 20
        _, odds_ratio, *_ = bool_measure.test_different(s, s, error_control=ec_i(0.05))
        assert odds_ratio == pytest.approx(
            1.0
        ), f"Expected OR=1 for identical samples, got {odds_ratio}"

    def test_or_large_effect_for_opposite_samples(self, bool_measure):
        """Completely opposite boolean samples should yield extreme OR."""
        s1 = [True] * 50 + [False] * 5
        s2 = [False] * 50 + [True] * 5
        _, odds_ratio, *_ = bool_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert (
            odds_ratio > 10 or odds_ratio < 0.1
        ), f"Expected extreme OR for opposite samples, got {odds_ratio}"

    def test_or_reciprocal_on_swap(self, bool_measure):
        """Swapping samples: OR_forward ≈ 1 / OR_reverse."""
        random.seed(7)
        s1 = [random.choice([True, False]) for _ in range(100)]
        s2 = [True] * 70 + [False] * 30
        _, or_fwd, *_ = bool_measure.test_different(s1, s2, error_control=ec_i(0.05))
        _, or_rev, *_ = bool_measure.test_different(s2, s1, error_control=ec_i(0.05))
        if (
            or_fwd > 0
            and or_rev > 0
            and math.isfinite(or_fwd)
            and math.isfinite(or_rev)
        ):
            assert or_fwd * or_rev == pytest.approx(
                1.0, abs=1e-6
            ), f"OR reciprocal violated: {or_fwd} * {or_rev} != 1"

    def test_or_ci_contains_point_estimate(self, bool_measure):
        """The CI should contain the odds ratio point estimate."""
        random.seed(15)
        s1 = [random.choice([True, False]) for _ in range(100)]
        s2 = [random.choice([True, False]) for _ in range(100)]
        _, odds_ratio, (lo, hi), *_ = bool_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert (
            lo <= odds_ratio <= hi
        ), f"CI [{lo}, {hi}] does not contain OR={odds_ratio}"

    def test_or_ci_lower_le_upper(self, bool_measure):
        """CI lower bound must be <= upper bound."""
        random.seed(88)
        s1 = [random.choice([True, False]) for _ in range(60)]
        s2 = [random.choice([True, False]) for _ in range(60)]
        _, _, (lo, hi), *_ = bool_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert lo <= hi

    def test_or_ci_with_zero_cell(self, bool_measure):
        """When a contingency table cell is zero, CI should be (0, inf)."""
        s1 = [True] * 50
        s2 = [False] * 50
        _, _, (lo, hi), *_ = bool_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert lo == 0.0
        assert hi == math.inf

    def test_accepts_binary_int_samples(self, bool_measure):
        """Binary int inputs (0/1) should be handled like booleans."""
        s1 = [1] * 70 + [0] * 30
        s2 = [1] * 50 + [0] * 50
        p, odds_ratio, ci, *_ = bool_measure.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert 0 <= p <= 1
        assert odds_ratio >= 0
        assert len(ci) == 2

    def test_rejects_non_binary_int_samples(self, bool_measure):
        """Only binary ints are valid for BooleanMeasure.test_different."""
        with pytest.raises(ValueError):
            bool_measure.test_different([0, 1, 2], [0, 1, 1])

    def test_sensitivity_changes_test_different_result(self):
        """Sensitivity should influence adjusted contingency counts and OR."""
        s1 = [1] * 130 + [0] * 70
        s2 = [1] * 100 + [0] * 100

        m_high_se = BooleanMeasure(
            name="b_hi_se",
            absolute_error=0.05,
            sensitivity=1.0,
            specificity=1.0,
        )
        m_low_se = BooleanMeasure(
            name="b_lo_se",
            absolute_error=0.05,
            sensitivity=0.9,
            specificity=1.0,
        )

        _, or_high_se, *_ = m_high_se.test_different(s1, s2)
        _, or_low_se, *_ = m_low_se.test_different(s1, s2)

        assert or_high_se != pytest.approx(or_low_se)

    def test_specificity_changes_test_different_result(self):
        """Specificity should influence adjusted contingency counts and OR."""
        s1 = [1] * 130 + [0] * 70
        s2 = [1] * 100 + [0] * 100

        m_baseline = BooleanMeasure(
            name="b_base",
            absolute_error=0.05,
            sensitivity=1.0,
            specificity=1.0,
        )
        m_adjusted = BooleanMeasure(
            name="b_adj",
            absolute_error=0.05,
            sensitivity=1.0,
            specificity=0.9,
        )

        _, or_baseline, *_ = m_baseline.test_different(s1, s2)
        _, or_adjusted, *_ = m_adjusted.test_different(s1, s2)

        assert or_baseline != pytest.approx(or_adjusted)


# =========================================================================
# 5c. test_different – TYPE_II CI properties
# =========================================================================


class TestTestDifferentTypeII:
    """Properties of the TYPE_II (power-focused) confidence interval."""

    @pytest.fixture(params=["boolean", "mean_known", "rank"])
    def any_testable(self, request):
        factories = {
            "boolean": lambda: BooleanMeasure(name="b", absolute_error=0.05),
            "mean_known": lambda: MeanMeasure(name="m", std=1.0, absolute_error=0.1),
            "rank": lambda: RankMeasure(name="r", max_rank=10, absolute_error=0.5),
        }
        return factories[request.param]()

    def test_type_ii_ci_narrower_than_type_i(self, any_testable):
        """TYPE_II uses one-sided z_β < two-sided z_{α/2}, so CI is narrower."""
        random.seed(42)
        if isinstance(any_testable, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(80)]
            s2 = [random.choice([True, False]) for _ in range(80)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(80)]
            s2 = [random.gauss(1, 1) for _ in range(80)]

        # Skip zero-cell case for BooleanMeasure (CI becomes (0, inf))
        if isinstance(any_testable, BooleanMeasure):
            if 0 in (sum(s1), len(s1) - sum(s1), sum(s2), len(s2) - sum(s2)):
                pytest.skip("zero-cell contingency table")

        _, _, (lo_i, hi_i), *_ = any_testable.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        _, _, (lo_ii, hi_ii), *_ = any_testable.test_different(
            s1, s2, error_control=ec_ii(0.05)
        )
        assert (hi_ii - lo_ii) <= (hi_i - lo_i)

    def test_type_ii_ci_lower_le_upper(self, any_testable):
        """TYPE_II CI bounds must be ordered."""
        random.seed(7)
        if isinstance(any_testable, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(60)]
            s2 = [random.choice([True, False]) for _ in range(60)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(60)]
            s2 = [random.gauss(0, 1) for _ in range(60)]
        _, _, (lo, hi), *_ = any_testable.test_different(
            s1, s2, error_control=ec_ii(0.05)
        )
        assert lo <= hi

    def test_type_ii_p_value_unchanged(self, any_testable):
        """The p-value should be the same regardless of the chosen error control."""
        random.seed(13)
        if isinstance(any_testable, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(50)]
            s2 = [random.choice([True, False]) for _ in range(50)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(50)]
            s2 = [random.gauss(1, 1) for _ in range(50)]
        p_i, *_ = any_testable.test_different(s1, s2, error_control=ec_i(0.05))
        p_ii, *_ = any_testable.test_different(s1, s2, error_control=ec_ii(0.05))
        assert p_i == pytest.approx(p_ii)


# =========================================================================
# 6. CategoricalMeasures – factory function properties
# =========================================================================


class TestCategoricalMeasures:
    def test_returns_correct_number_of_measures(self):
        measures = CategoricalMeasures("cat", categories=5, absolute_error=0.05)
        assert len(measures) == 5

    def test_each_element_is_boolean_measure(self):
        measures = CategoricalMeasures("cat", categories=3, absolute_error=0.05)
        for m in measures:
            assert isinstance(m, BooleanMeasure)

    def test_names_are_unique(self):
        measures = CategoricalMeasures("cat", categories=4, absolute_error=0.05)
        names = [m.name for m in measures]
        assert len(set(names)) == len(names)

    def test_all_share_same_parameters(self):
        measures = CategoricalMeasures(
            "cat", categories=3, std=0.3, absolute_error=0.02, repeats=2
        )
        for m in measures:
            assert m.std == 0.3
            assert m.absolute_error == 0.02
            assert m.repeats == 2

    def test_single_category(self):
        measures = CategoricalMeasures("cat", categories=1, absolute_error=0.05)
        assert len(measures) == 1
        assert isinstance(measures[0], BooleanMeasure)


# =========================================================================
# 7. Measure identity / hashing
# =========================================================================


class TestMeasureHashing:
    def test_same_name_same_hash(self):
        a = BooleanMeasure(name="x", absolute_error=0.05)
        b = BooleanMeasure(name="x", absolute_error=0.10)
        assert hash(a) == hash(b)

    def test_different_name_likely_different_hash(self):
        a = BooleanMeasure(name="x", absolute_error=0.05)
        b = BooleanMeasure(name="y", absolute_error=0.05)
        assert hash(a) != hash(b)

    def test_usable_in_set(self):
        measures = {
            BooleanMeasure(name="a", absolute_error=0.05),
            MeanMeasure(name="b", std=1.0, absolute_error=0.1),
        }
        assert len(measures) == 2

    def test_different_types_same_name_same_hash(self):
        a = BooleanMeasure(name="x", absolute_error=0.05)
        b = MeanMeasure(name="x", std=1.0, absolute_error=0.1)
        assert hash(a) == hash(b)


# =========================================================================
# 8. Consistency: sample_size <-> confidence round-trip
# =========================================================================


class TestRoundTrip:
    """If we compute n for a target confidence, then compute_error_probability(n) should
    return at least that target (since n is rounded up)."""

    @pytest.mark.parametrize("target_conf", [0.80, 0.90, 0.95, 0.99])
    def test_boolean(self, target_conf):
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(ec_i(1 - target_conf))
        achieved = m.compute_error_probability(n, ec_i())
        assert (
            achieved >= target_conf - 0.02
        ), f"target={target_conf}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_conf", [0.80, 0.90, 0.95, 0.99])
    def test_variance(self, target_conf):
        m = VarianceMeasure(name="v", relative_error=0.1)
        n = m.compute_sample_size(ec_i(1 - target_conf))
        achieved = m.compute_error_probability(n, ec_i())
        assert (
            achieved >= target_conf - 0.02
        ), f"target={target_conf}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_conf", [0.80, 0.90, 0.95, 0.99])
    def test_mean_known_std(self, target_conf):
        m = MeanMeasure(name="m", std=1.0, absolute_error=0.1)
        n = m.compute_sample_size(ec_i(1 - target_conf))
        achieved = m.compute_error_probability(n, ec_i())
        assert (
            achieved >= target_conf - 0.02
        ), f"target={target_conf}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_power", [0.80, 0.90, 0.95, 0.99])
    def test_boolean_type_ii(self, target_power):
        """TYPE_II round-trip: sample size for target power should achieve at least that power."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(ec_ii(1 - target_power))
        achieved = m.compute_error_probability(n, ec_ii(1 - target_power))
        assert (
            achieved >= target_power - 0.02
        ), f"target_power={target_power}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_power", [0.80, 0.90, 0.95, 0.99])
    def test_variance_type_ii(self, target_power):
        """TYPE_II round-trip for VarianceMeasure."""
        m = VarianceMeasure(name="v", relative_error=0.1)
        n = m.compute_sample_size(ec_ii(1 - target_power))
        achieved = m.compute_error_probability(n, ec_ii(1 - target_power))
        assert (
            achieved >= target_power - 0.02
        ), f"target_power={target_power}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_power", [0.80, 0.90, 0.95, 0.99])
    def test_mean_known_std_type_ii(self, target_power):
        """TYPE_II round-trip for MeanMeasure with known std."""
        m = MeanMeasure(name="m", std=1.0, absolute_error=0.1)
        n = m.compute_sample_size(ec_ii(1 - target_power))
        achieved = m.compute_error_probability(n, ec_ii(1 - target_power))
        assert (
            achieved >= target_power - 0.02
        ), f"target_power={target_power}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_power", [0.80, 0.90, 0.95, 0.99])
    def test_rank_type_ii(self, target_power):
        """TYPE_II round-trip for RankMeasure."""
        m = RankMeasure(name="r", max_rank=10, absolute_error=0.5)
        n = m.compute_sample_size(ec_ii(1 - target_power))
        achieved = m.compute_error_probability(n, ec_ii(1 - target_power))
        assert (
            achieved >= target_power - 0.02
        ), f"target_power={target_power}, n={n}, achieved={achieved}"


# =========================================================================
# 9. Edge cases
# =========================================================================


class TestEdgeCases:
    def test_very_low_confidence(self):
        """Even very low confidence (high error) should produce a valid (small) sample size."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(error_control=ec_i(0.50))
        assert n >= 1

    def test_very_high_confidence(self):
        """Very high confidence (very low error) should still return a finite sample size."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(error_control=ec_i(0.001))
        assert n > 0 and math.isfinite(n)

    def test_large_absolute_error_small_sample(self):
        """Large error tolerance should need very few samples."""
        m = BooleanMeasure(name="b", absolute_error=0.5)
        n = m.compute_sample_size(error_control=ec_i(0.05))
        # With 50% error tolerance on a boolean, we need very few samples
        assert n < 100

    def test_boolean_test_different_all_same_class(self):
        """When both samples are all True, they should not be different."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        s = [True] * 50
        p, *_ = m.test_different(s, s, error_control=ec_i(0.05))
        assert p >= 0.05

    def test_mean_test_different_constant_samples(self):
        """Two constant samples with same value should not be flagged different."""
        m = MeanMeasure(name="m", std=1.0, absolute_error=0.1)
        s = [5.0] * 50
        p, *_ = m.test_different(s, s, error_control=ec_i(0.05))
        assert p >= 0.05


# =========================================================================
# 10. test_different – Type I and Type II error return values
# =========================================================================


class TestTestDifferentErrorRates:
    """Properties of the Type I and Type II error values returned by test_different."""

    @pytest.fixture(params=["boolean", "mean_known", "rank"])
    def any_testable(self, request):
        factories = {
            "boolean": lambda: BooleanMeasure(name="b", absolute_error=0.05),
            "mean_known": lambda: MeanMeasure(name="m", std=1.0, absolute_error=0.1),
            "rank": lambda: RankMeasure(name="r", max_rank=10, absolute_error=0.5),
        }
        return factories[request.param]()

    def test_errors_in_valid_range(self, any_testable):
        """Type I and Type II errors must be in [0, 1]."""
        random.seed(42)
        s1 = [random.gauss(0, 1) for _ in range(60)]
        s2 = [random.gauss(1, 1) for _ in range(60)]
        if isinstance(any_testable, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(60)]
            s2 = [random.choice([True, False]) for _ in range(60)]
        elif isinstance(any_testable, RankMeasure):
            s1 = [random.randint(1, 10) for _ in range(60)]
            s2 = [random.randint(1, 10) for _ in range(60)]
        _, _, _, type_i_error, type_ii_error = any_testable.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert 0 <= type_i_error <= 1
        assert 0 <= type_ii_error <= 1

    def test_type_i_error_decreases_with_sample_size(self):
        """Larger samples should yield a smaller Type I error."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        random.seed(1)
        s1_small = [random.choice([True, False]) for _ in range(30)]
        s2_small = [random.choice([True, False]) for _ in range(30)]
        s1_large = [random.choice([True, False]) for _ in range(300)]
        s2_large = [random.choice([True, False]) for _ in range(300)]
        _, _, _, type_i_small, _ = m.test_different(
            s1_small, s2_small, error_control=ec_i(0.05)
        )
        _, _, _, type_i_large, _ = m.test_different(
            s1_large, s2_large, error_control=ec_i(0.05)
        )
        assert type_i_large <= type_i_small

    def test_type_ii_error_decreases_with_sample_size(self):
        """Larger samples should yield a smaller Type II error (more power)."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        random.seed(2)
        s1_small = [random.choice([True, False]) for _ in range(30)]
        s2_small = [random.choice([True, False]) for _ in range(30)]
        s1_large = [random.choice([True, False]) for _ in range(300)]
        s2_large = [random.choice([True, False]) for _ in range(300)]
        _, _, _, _, type_ii_small = m.test_different(
            s1_small, s2_small, error_control=ec_i(0.05)
        )
        _, _, _, _, type_ii_large = m.test_different(
            s1_large, s2_large, error_control=ec_i(0.05)
        )
        assert type_ii_large <= type_ii_small

    def test_type_ii_error_greater_than_type_i_error(self, any_testable):
        """At a sample size calibrated through the same margin, Type II error is larger
        because power must also clear the test threshold α."""
        random.seed(99)
        if isinstance(any_testable, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(80)]
            s2 = [random.choice([True, False]) for _ in range(80)]
        elif isinstance(any_testable, RankMeasure):
            s1 = [random.randint(1, 10) for _ in range(80)]
            s2 = [random.randint(1, 10) for _ in range(80)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(80)]
            s2 = [random.gauss(1, 1) for _ in range(80)]
        _, _, _, type_i_error, type_ii_error = any_testable.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        assert type_ii_error >= type_i_error, (
            f"{type(any_testable).__name__}: expected TYPE_II error >= TYPE_I error, "
            f"got {type_ii_error} < {type_i_error}"
        )

    def test_errors_consistent_with_compute_error_probability(self):
        """Type I error returned by test_different should equal
        1 - compute_error_probability(n, ec_i()) for the same n."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = 100
        s1 = [True] * (n // 2) + [False] * (n // 2)
        s2 = [True] * (n // 2) + [False] * (n // 2)
        _, _, _, type_i_error, type_ii_error = m.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        expected_type_i = 1 - m.compute_error_probability(n, ec_i())
        expected_type_ii = 1 - m.compute_error_probability(n, ec_ii())
        assert type_i_error == pytest.approx(expected_type_i)
        assert type_ii_error == pytest.approx(expected_type_ii)

    def test_errors_independent_of_interval_error_control(self, any_testable):
        """The error rates returned are properties of the sample size,
        not of the interval error control object (which only affects the CI)."""
        random.seed(7)
        if isinstance(any_testable, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(60)]
            s2 = [random.choice([True, False]) for _ in range(60)]
        elif isinstance(any_testable, RankMeasure):
            s1 = [random.randint(1, 10) for _ in range(60)]
            s2 = [random.randint(1, 10) for _ in range(60)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(60)]
            s2 = [random.gauss(0, 1) for _ in range(60)]
        _, _, _, t1_from_i, t2_from_i = any_testable.test_different(
            s1, s2, error_control=ec_i(0.05)
        )
        _, _, _, t1_from_ii, t2_from_ii = any_testable.test_different(
            s1, s2, error_control=ec_ii(0.05)
        )
        assert t1_from_i == pytest.approx(t1_from_ii)
        assert t2_from_i == pytest.approx(t2_from_ii)
