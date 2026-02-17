import math
import random

import pytest

from reeval.measures.boolean_measure import BooleanMeasure, CategoricalMeasures
from reeval.measures.mean_measure import MeanMeasure
from reeval.measures.variance_measure import VarianceMeasure
from reeval.measures.rank_measure import RankMeasure


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
        n = any_measure.compute_sample_size(confidence=0.95)
        assert isinstance(n, (int, float))
        assert n > 0

    def test_monotone_increasing_with_confidence(self, any_measure):
        """Higher confidence must require at least as many samples."""
        n_low = any_measure.compute_sample_size(confidence=0.80)
        n_high = any_measure.compute_sample_size(confidence=0.99)
        assert n_high >= n_low

    def test_increases_with_repetition_multiplier(self, any_measure):
        """More simultaneous comparisons (Bonferroni) should not decrease the
        required sample size."""
        n1 = any_measure.compute_sample_size(confidence=0.95, repetition_multiplier=1)
        n2 = any_measure.compute_sample_size(confidence=0.95, repetition_multiplier=5)
        assert n2 >= n1

    def test_increases_with_repeats(self):
        """More built-in repeats (Bonferroni) should not decrease sample size."""
        m1 = BooleanMeasure(name="b", repeats=1, absolute_error=0.05)
        m2 = BooleanMeasure(name="b", repeats=10, absolute_error=0.05)
        assert m2.compute_sample_size(0.95) >= m1.compute_sample_size(0.95)


class TestSampleSizeDecreasesWithTolerance:
    """Larger error tolerance should require fewer samples."""

    def test_boolean_measure(self):
        m_tight = BooleanMeasure(name="b", absolute_error=0.01)
        m_loose = BooleanMeasure(name="b", absolute_error=0.10)
        assert m_tight.compute_sample_size(0.95) > m_loose.compute_sample_size(0.95)

    def test_mean_measure_known_std(self):
        m_tight = MeanMeasure(name="m", std=1.0, absolute_error=0.01)
        m_loose = MeanMeasure(name="m", std=1.0, absolute_error=0.10)
        assert m_tight.compute_sample_size(0.95) > m_loose.compute_sample_size(0.95)

    def test_variance_measure(self):
        m_tight = VarianceMeasure(name="v", relative_error=0.01)
        m_loose = VarianceMeasure(name="v", relative_error=0.10)
        assert m_tight.compute_sample_size(0.95) > m_loose.compute_sample_size(0.95)

    @pytest.mark.xfail(reason="RankMeasure.std uses self.k instead of self.max_rank")
    def test_rank_measure(self):
        m_tight = RankMeasure(name="r", max_rank=10, absolute_error=0.1)
        m_loose = RankMeasure(name="r", max_rank=10, absolute_error=1.0)
        assert m_tight.compute_sample_size(0.95) > m_loose.compute_sample_size(0.95)

    @pytest.mark.xfail(reason="student_sample_size does not converge (infinite loop)")
    def test_mean_measure_unknown_std(self):
        m_tight = MeanMeasure(name="m", absolute_error=0.01)
        m_loose = MeanMeasure(name="m", absolute_error=0.10)
        assert m_tight.compute_sample_size(0.95) > m_loose.compute_sample_size(0.95)


class TestSampleSizeIncreasesWithUncertainty:
    """More uncertain distributions need more samples."""

    def test_boolean_higher_std_needs_more_samples(self):
        m_low = BooleanMeasure(name="b", std=0.1, absolute_error=0.05)
        m_high = BooleanMeasure(name="b", std=0.5, absolute_error=0.05)
        assert m_high.compute_sample_size(0.95) >= m_low.compute_sample_size(0.95)

    def test_mean_higher_std_needs_more_samples(self):
        m_low = MeanMeasure(name="m", std=0.5, absolute_error=0.1)
        m_high = MeanMeasure(name="m", std=2.0, absolute_error=0.1)
        assert m_high.compute_sample_size(0.95) >= m_low.compute_sample_size(0.95)

    @pytest.mark.xfail(reason="RankMeasure.std uses self.k instead of self.max_rank")
    def test_rank_higher_max_rank_needs_more_samples(self):
        m_small = RankMeasure(name="r", max_rank=3, absolute_error=0.5)
        m_large = RankMeasure(name="r", max_rank=20, absolute_error=0.5)
        assert m_large.compute_sample_size(0.95) >= m_small.compute_sample_size(0.95)


# =========================================================================
# 2. compute_confidence – general properties
# =========================================================================


class TestComputeConfidence:
    def test_returns_value_in_valid_range(self, any_measure):
        """Confidence should be in (0, 1] for reasonable sample sizes."""
        n = any_measure.compute_sample_size(confidence=0.95)
        conf = any_measure.compute_confidence(n)
        assert 0 < conf <= 1.0

    def test_monotone_increasing_with_sample_size(self, any_measure):
        """More samples should not decrease confidence."""
        n_base = any_measure.compute_sample_size(confidence=0.90)
        conf_small = any_measure.compute_confidence(max(n_base, 10))
        conf_large = any_measure.compute_confidence(max(n_base * 5, 50))
        assert conf_large >= conf_small

    def test_more_repeats_reduce_confidence(self):
        """Bonferroni correction: more repeats at same sample size -> lower confidence."""
        m1 = BooleanMeasure(name="b", repeats=1, absolute_error=0.05)
        m2 = BooleanMeasure(name="b", repeats=10, absolute_error=0.05)
        n = 500
        assert m1.compute_confidence(n) >= m2.compute_confidence(n)


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
        err = error_measure.compute_absolute_error(sample_size=100, confidence=0.95)
        assert err > 0

    def test_decreases_with_sample_size(self, error_measure):
        err_small = error_measure.compute_absolute_error(
            sample_size=50, confidence=0.95
        )
        err_large = error_measure.compute_absolute_error(
            sample_size=500, confidence=0.95
        )
        assert err_large < err_small

    def test_increases_with_confidence(self, error_measure):
        """At the same sample size, higher confidence -> wider error bar."""
        err_low = error_measure.compute_absolute_error(sample_size=200, confidence=0.80)
        err_high = error_measure.compute_absolute_error(
            sample_size=200, confidence=0.99
        )
        assert err_high >= err_low

    def test_variance_measure_raises(self, variance_measure):
        with pytest.raises(NotImplementedError):
            variance_measure.compute_absolute_error(sample_size=100, confidence=0.95)

    @pytest.mark.xfail(reason="RankMeasure.std uses self.k instead of self.max_rank")
    def test_rank_absolute_error_decreases_with_sample_size(self):
        m = RankMeasure(name="rank", max_rank=10, absolute_error=0.5)
        err_small = m.compute_absolute_error(sample_size=50, confidence=0.95)
        err_large = m.compute_absolute_error(sample_size=500, confidence=0.95)
        assert err_large < err_small


# =========================================================================
# 4. VarianceMeasure.compute_relative_error – general properties
# =========================================================================


class TestVarianceRelativeError:
    def test_returns_positive(self, variance_measure):
        err = variance_measure.compute_relative_error(sample_size=100, confidence=0.95)
        assert err > 0

    def test_decreases_with_sample_size(self, variance_measure):
        err_small = variance_measure.compute_relative_error(
            sample_size=50, confidence=0.95
        )
        err_large = variance_measure.compute_relative_error(
            sample_size=500, confidence=0.95
        )
        assert abs(err_large) < abs(err_small)

    def test_increases_with_confidence(self, variance_measure):
        """At the same sample size, higher confidence -> wider relative error."""
        err_low = variance_measure.compute_relative_error(
            sample_size=200, confidence=0.80
        )
        err_high = variance_measure.compute_relative_error(
            sample_size=200, confidence=0.99
        )
        assert abs(err_high) >= abs(err_low)


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
        p, effect, ci = testable_measure.test_different(s1, s2)
        assert 0 <= p <= 1

    def test_identical_samples_high_p_value(self, testable_measure):
        """Two identical samples should not be flagged as different."""
        random.seed(123)
        if isinstance(testable_measure, BooleanMeasure):
            s = [random.choice([True, False]) for _ in range(100)]
        else:
            s = [random.gauss(5, 1) for _ in range(100)]
        p, _, _ = testable_measure.test_different(s, s)
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
        p, _, _ = testable_measure.test_different(s1, s2)
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
        result = testable_measure.test_different(s1, s2)
        assert len(result) == 3
        p_value, effect_size, ci = result
        assert isinstance(p_value, float)
        assert isinstance(effect_size, float)
        assert isinstance(ci, tuple) and len(ci) == 2

    def test_ci_lower_le_upper(self, testable_measure):
        """Confidence interval lower bound should be <= upper bound."""
        random.seed(7)
        if isinstance(testable_measure, BooleanMeasure):
            s1 = [random.choice([True, False]) for _ in range(60)]
            s2 = [random.choice([True, False]) for _ in range(60)]
        else:
            s1 = [random.gauss(0, 1) for _ in range(60)]
            s2 = [random.gauss(0, 1) for _ in range(60)]
        _, _, (lo, hi) = testable_measure.test_different(s1, s2)
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
        p_forward, _, _ = testable_measure.test_different(s1, s2)
        p_reverse, _, _ = testable_measure.test_different(s2, s1)
        assert abs(p_forward - p_reverse) < 1e-10

    @pytest.mark.xfail(reason="RankMeasure.std uses self.k instead of self.max_rank")
    def test_rank_test_different_p_in_range(self):
        m = RankMeasure(name="rank", max_rank=10, absolute_error=0.5)
        random.seed(42)
        s1 = [random.randint(1, 10) for _ in range(50)]
        s2 = [random.randint(1, 10) for _ in range(50)]
        p, _, _ = m.test_different(s1, s2)
        assert 0 <= p <= 1


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
    """If we compute n for a target confidence, then compute_confidence(n) should
    return at least that target (since n is rounded up)."""

    @pytest.mark.parametrize("target_conf", [0.80, 0.90, 0.95, 0.99])
    def test_boolean(self, target_conf):
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(target_conf)
        achieved = m.compute_confidence(n)
        assert (
            achieved >= target_conf - 0.02
        ), f"target={target_conf}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_conf", [0.80, 0.90, 0.95, 0.99])
    def test_variance(self, target_conf):
        m = VarianceMeasure(name="v", relative_error=0.1)
        n = m.compute_sample_size(target_conf)
        achieved = m.compute_confidence(n)
        assert (
            achieved >= target_conf - 0.02
        ), f"target={target_conf}, n={n}, achieved={achieved}"

    @pytest.mark.parametrize("target_conf", [0.80, 0.90, 0.95, 0.99])
    def test_mean_known_std(self, target_conf):
        m = MeanMeasure(name="m", std=1.0, absolute_error=0.1)
        n = m.compute_sample_size(target_conf)
        achieved = m.compute_confidence(n)
        assert (
            achieved >= target_conf - 0.02
        ), f"target={target_conf}, n={n}, achieved={achieved}"


# =========================================================================
# 9. Edge cases
# =========================================================================


class TestEdgeCases:
    def test_very_low_confidence(self):
        """Even very low confidence should produce a valid (small) sample size."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(confidence=0.50)
        assert n >= 1

    def test_very_high_confidence(self):
        """Very high confidence should still return a finite sample size."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        n = m.compute_sample_size(confidence=0.999)
        assert n > 0 and math.isfinite(n)

    def test_large_absolute_error_small_sample(self):
        """Large error tolerance should need very few samples."""
        m = BooleanMeasure(name="b", absolute_error=0.5)
        n = m.compute_sample_size(confidence=0.95)
        # With 50% error tolerance on a boolean, we need very few samples
        assert n < 100

    def test_boolean_test_different_all_same_class(self):
        """When both samples are all True, they should not be different."""
        m = BooleanMeasure(name="b", absolute_error=0.05)
        s = [True] * 50
        p, _, _ = m.test_different(s, s)
        assert p >= 0.05

    def test_mean_test_different_constant_samples(self):
        """Two constant samples with same value should not be flagged different."""
        m = MeanMeasure(name="m", std=1.0, absolute_error=0.1)
        s = [5.0] * 50
        p, _, _ = m.test_different(s, s)
        assert p >= 0.05
