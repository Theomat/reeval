import math
import pytest
from scipy import stats

from reeval.measure import Measure, MeasureType


class TestMeasurePostInit:
    """Tests for Measure.__post_init__ automatic value_range initialization."""

    def test_boolean_proportion_sets_value_range(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
        )
        assert measure.value_range == (0, 1)

    def test_categorical_proportion_sets_value_range(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_CATEGORICAL,
            categories=3,
        )
        assert measure.value_range == (0, 1)

    def test_mean_does_not_auto_set_value_range(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.MEAN,
        )
        assert measure.value_range is None

    def test_variance_does_not_auto_set_value_range(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.VARIANCE,
        )
        assert measure.value_range is None

    def test_explicit_value_range_preserved(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.MEAN,
            value_range=(10, 50),
        )
        assert measure.value_range == (10, 50)


class TestGetAdjustedRepetitions:
    """Tests for Measure._get_adjusted_repetitions_."""

    def test_non_categorical_returns_repetitions(self, boolean_proportion_measure):
        assert boolean_proportion_measure._get_adjusted_repetitions_() == 1

    def test_non_categorical_with_multiple_repetitions(self, measure_with_repetitions):
        assert measure_with_repetitions._get_adjusted_repetitions_() == 3

    def test_categorical_multiplies_by_categories(self, categorical_proportion_measure):
        # categories=5, repetitions=1 (default)
        assert categorical_proportion_measure._get_adjusted_repetitions_() == 5

    def test_categorical_with_repetitions(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_CATEGORICAL,
            categories=4,
            repetitions=2,
        )
        assert measure._get_adjusted_repetitions_() == 8

    def test_mean_returns_repetitions(self, mean_measure):
        assert mean_measure._get_adjusted_repetitions_() == 1

    def test_variance_returns_repetitions(self, variance_measure):
        assert variance_measure._get_adjusted_repetitions_() == 1


class TestComputeAdjustedZ:
    """Tests for Measure._compute_adjusted_z_."""

    def test_basic_z_score_calculation(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
            std=0.5,
        )
        z = measure._compute_adjusted_z_(0.95, "test")
        # For 95% confidence, z ~ 1.96 (two-tailed: 1 - alpha/2 = 0.975)
        expected_z = stats.Normal().icdf(0.975) * 0.5
        assert math.isclose(z, expected_z, rel_tol=1e-6)

    def test_auto_fills_std_from_value_range(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.MEAN,
            value_range=(0, 100),
        )
        assert measure.std is None
        measure._compute_adjusted_z_(0.95, "test")
        assert measure.std == 25.0  # (100 - 0) / 4

    def test_confidence_adjustment_with_repetitions(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
            std=0.5,
            repetitions=3,
        )
        z = measure._compute_adjusted_z_(0.95, "test")
        # With repetitions, alpha is adjusted using Sickhart's formula
        alpha = 1 - 0.95
        adjusted_alpha = 1 - math.pow(1 - alpha, 1 / 3)
        # Two-tailed: use 1 - adjusted_alpha / 2
        expected_z = stats.Normal().icdf(1 - adjusted_alpha / 2) * 0.5
        assert math.isclose(z, expected_z, rel_tol=1e-6)

    def test_variance_skips_std_multiplication(self, variance_measure):
        z = variance_measure._compute_adjusted_z_(0.95, "test")
        # For variance, z is not multiplied by std (two-tailed)
        expected_z = stats.Normal().icdf(0.975)
        assert math.isclose(z, expected_z, rel_tol=1e-6)

    def test_raises_for_confidence_below_zero(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
        )
        with pytest.raises(AssertionError, match="confidence must be in"):
            measure._compute_adjusted_z_(-0.1, "test")

    def test_raises_for_confidence_above_one(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
        )
        with pytest.raises(AssertionError, match="confidence must be in"):
            measure._compute_adjusted_z_(1.5, "test")

    def test_raises_without_std_or_value_range_for_non_variance(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.MEAN,
        )
        with pytest.raises(
            AssertionError, match="std or value_range must be specified"
        ):
            measure._compute_adjusted_z_(0.95, "sample size")


class TestComputeSampleSize:
    """Tests for Measure.compute_sample_size."""

    def test_computes_correct_sample_size(self, boolean_proportion_measure):
        sample_size = boolean_proportion_measure.compute_sample_size(0.95)
        # For boolean proportion: value_range=(0,1), std=0.5 (range / 2)
        # z = Normal().icdf(0.975) * 0.5 (two-tailed)
        # sample_size = ceil((z / 0.05)^2)
        z = stats.Normal().icdf(0.975) * 0.5
        expected = int(math.ceil((z / 0.05) ** 2))
        assert sample_size == expected

    def test_computes_sample_size_with_explicit_std(self, mean_measure):
        sample_size = mean_measure.compute_sample_size(0.95)
        z = stats.Normal().icdf(0.975) * 2.0  # std=2.0, two-tailed
        expected = int(math.ceil((z / 0.1) ** 2))
        assert sample_size == expected

    def test_raises_without_absolute_error(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
        )
        with pytest.raises(AssertionError, match="absolute_error must be specified"):
            measure.compute_sample_size(0.95)

    def test_sample_size_increases_with_confidence(self, boolean_proportion_measure):
        size_90 = boolean_proportion_measure.compute_sample_size(0.90)
        size_95 = boolean_proportion_measure.compute_sample_size(0.95)
        size_99 = boolean_proportion_measure.compute_sample_size(0.99)
        assert size_90 < size_95 < size_99

    def test_sample_size_with_repetitions(self, measure_with_repetitions):
        sample_size = measure_with_repetitions.compute_sample_size(0.95)
        # With repetitions, the confidence is adjusted
        alpha = 1 - 0.95
        adjusted_alpha = 1 - math.pow(1 - alpha, 1 / 3)
        # Two-tailed: use 1 - adjusted_alpha / 2, std=0.5 for boolean proportion
        z = stats.Normal().icdf(1 - adjusted_alpha / 2) * 0.5
        expected = int(math.ceil((z / 0.05) ** 2))
        assert sample_size == expected


class TestComputeAbsoluteError:
    """Tests for Measure.compute_absolute_error."""

    def test_computes_correct_error(self, boolean_proportion_measure):
        error = boolean_proportion_measure.compute_absolute_error(1000, 0.95)
        # Two-tailed z, std=0.5 for boolean proportion
        z = stats.Normal().icdf(0.975) * 0.5
        expected = z / math.sqrt(1000)
        assert math.isclose(error, expected, rel_tol=1e-6)

    def test_error_decreases_with_sample_size(self, boolean_proportion_measure):
        error_100 = boolean_proportion_measure.compute_absolute_error(100, 0.95)
        error_1000 = boolean_proportion_measure.compute_absolute_error(1000, 0.95)
        error_10000 = boolean_proportion_measure.compute_absolute_error(10000, 0.95)
        assert error_100 > error_1000 > error_10000

    def test_error_increases_with_confidence(self, boolean_proportion_measure):
        error_90 = boolean_proportion_measure.compute_absolute_error(1000, 0.90)
        error_95 = boolean_proportion_measure.compute_absolute_error(1000, 0.95)
        error_99 = boolean_proportion_measure.compute_absolute_error(1000, 0.99)
        assert error_90 < error_95 < error_99

    def test_variance_measure_error(self, variance_measure):
        error = variance_measure.compute_absolute_error(1000, 0.95)
        z = stats.Normal().icdf(0.975)  # No std multiplication for variance, two-tailed
        expected = z / math.sqrt(1000)
        assert math.isclose(error, expected, rel_tol=1e-6)


class TestComputeConfidence:
    """Tests for Measure.compute_confidence."""

    def test_computes_correct_confidence(self, boolean_proportion_measure):
        confidence = boolean_proportion_measure.compute_confidence(1000)
        # adjusted_sample_size = sqrt(1000) / 0.5 (std=0.5 for boolean proportion)
        # confidence = Normal().cdf(adjusted_sample_size * 0.05)
        adjusted = math.sqrt(1000) / 0.5
        expected = stats.Normal().cdf(adjusted * 0.05)
        assert math.isclose(confidence, expected, rel_tol=1e-6)

    def test_confidence_increases_with_sample_size(self, boolean_proportion_measure):
        conf_100 = boolean_proportion_measure.compute_confidence(100)
        conf_1000 = boolean_proportion_measure.compute_confidence(1000)
        conf_10000 = boolean_proportion_measure.compute_confidence(10000)
        assert conf_100 < conf_1000 < conf_10000

    def test_raises_without_absolute_error(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
        )
        with pytest.raises(AssertionError, match="absolute_error must be specified"):
            measure.compute_confidence(1000)

    def test_confidence_with_repetitions(self, measure_with_repetitions):
        confidence = measure_with_repetitions.compute_confidence(1000)
        # Base confidence calculation (std=0.5 for boolean proportion)
        adjusted = math.sqrt(1000) / 0.5
        base_conf = stats.Normal().cdf(adjusted * 0.05)
        # Adjust for repetitions
        alpha = 1 - base_conf
        true_alpha = 1 - (1 - alpha) ** 3
        expected = 1 - true_alpha
        assert math.isclose(confidence, expected, rel_tol=1e-6)

    def test_variance_confidence(self, variance_measure):
        confidence = variance_measure.compute_confidence(1000)
        # For variance, std is not used
        adjusted = math.sqrt(1000)
        expected = stats.Normal().cdf(adjusted * 0.1)
        assert math.isclose(confidence, expected, rel_tol=1e-6)

    def test_auto_fills_std_from_value_range(self, mean_measure_with_range):
        assert mean_measure_with_range.std is None
        mean_measure_with_range.compute_confidence(1000)
        assert mean_measure_with_range.std == 25.0  # (100 - 0) / 4
