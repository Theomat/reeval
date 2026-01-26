import math
import pytest

from reeval.measure import Measure, MeasureType
from reeval.evaluation import Evaluation
from reeval.population import FinitePopulation, InfinitePopulation


class TestGetTotalRepeats:
    """Tests for Evaluation._get_total_repeats_."""

    def test_single_measure_single_comparison(self, basic_evaluation):
        # Single boolean proportion measure, max_comparisons=1
        # adjusted_repetitions = 1, total = 1 * 1 = 1
        assert basic_evaluation._get_total_repeats_() == 1

    def test_single_measure_multiple_comparisons(self, boolean_proportion_measure):
        evaluation = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=3,
            confidence=0.95,
        )
        assert evaluation._get_total_repeats_() == 3

    def test_multiple_measures(self, boolean_proportion_measure, mean_measure):
        evaluation = Evaluation(
            measures=[boolean_proportion_measure, mean_measure],
            max_comparisons=2,
            confidence=0.95,
        )
        # Both measures have adjusted_repetitions=1, total = (1 + 1) * 2 = 4
        assert evaluation._get_total_repeats_() == 4

    def test_categorical_measure_counts_categories(
        self, categorical_proportion_measure
    ):
        evaluation = Evaluation(
            measures=[categorical_proportion_measure],
            max_comparisons=1,
            confidence=0.95,
        )
        # Categorical with 5 categories: adjusted_repetitions = 5
        assert evaluation._get_total_repeats_() == 5

    def test_measure_with_repetitions(self, measure_with_repetitions):
        evaluation = Evaluation(
            measures=[measure_with_repetitions],
            max_comparisons=2,
            confidence=0.95,
        )
        # repetitions=3, max_comparisons=2, total = 3 * 2 = 6
        assert evaluation._get_total_repeats_() == 6


class TestComputeSampleSize:
    """Tests for Evaluation.compute_sample_size."""

    def test_computes_sample_size_for_single_measure(self, basic_evaluation):
        sample_size = basic_evaluation.compute_sample_size()
        # Should compute sample size for the single measure
        assert sample_size > 0
        assert isinstance(sample_size, int)

    def test_returns_max_across_measures(self, multi_measure_evaluation):
        sample_size = multi_measure_evaluation.compute_sample_size()
        # Should return the maximum sample size required by any measure
        assert sample_size > 0

    def test_applies_finite_population_correction(self, finite_population_evaluation):
        # Get sample size with infinite population first
        infinite_eval = Evaluation(
            measures=finite_population_evaluation.measures,
            max_comparisons=1,
            confidence=0.95,
            population=InfinitePopulation(),
        )
        infinite_size = infinite_eval.compute_sample_size()
        finite_size = finite_population_evaluation.compute_sample_size()
        # Finite population correction should reduce required sample size
        assert finite_size <= infinite_size

    def test_finite_population_correction_formula(self):
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
            absolute_error=0.05,
        )
        pop_size = 500
        evaluation = Evaluation(
            measures=[measure],
            max_comparisons=1,
            confidence=0.95,
            population=FinitePopulation(size=pop_size),
        )
        # Compute with infinite population
        inf_eval = Evaluation(
            measures=[measure],
            max_comparisons=1,
            confidence=0.95,
        )
        n0 = inf_eval.compute_sample_size()
        # Apply Cohenn's formula manually
        expected = int(math.ceil(n0 / (1 + (n0 - 1) / pop_size)))
        assert evaluation.compute_sample_size() == expected

    def test_sample_size_increases_with_comparisons(self, boolean_proportion_measure):
        eval_1 = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=1,
            confidence=0.95,
        )
        eval_3 = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=3,
            confidence=0.95,
        )
        # More comparisons should require larger sample size
        assert eval_1.compute_sample_size() <= eval_3.compute_sample_size()


class TestGetAdjustedSampleSize:
    """Tests for Evaluation.__get_adjusted_sample_size__."""

    def test_returns_sample_size_for_infinite_population(
        self, evaluation_with_sample_size
    ):
        # Access the private method directly (double underscores at both ends = not mangled)
        adjusted = evaluation_with_sample_size.__get_adjusted_sample_size__()
        assert adjusted == 500

    def test_applies_finite_population_adjustment(self, boolean_proportion_measure):
        pop_size = 1000
        sample_size = 200
        evaluation = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=1,
            sample_size=sample_size,
            population=FinitePopulation(size=pop_size),
        )
        adjusted = evaluation.__get_adjusted_sample_size__()
        # Formula: n * (N-1) / (N-n)
        expected = sample_size * (pop_size - 1) / (pop_size - sample_size)
        assert math.isclose(adjusted, expected, rel_tol=1e-6)

    def test_raises_without_sample_size(self, basic_evaluation):
        with pytest.raises(AssertionError, match="sample size must be specified"):
            basic_evaluation.__get_adjusted_sample_size__()


class TestComputeConfidences:
    """Tests for Evaluation.compute_confidences."""

    def test_returns_total_and_individual_confidences(
        self, evaluation_with_sample_size
    ):
        total_conf, confs = evaluation_with_sample_size.compute_confidences()
        assert isinstance(total_conf, float)
        assert isinstance(confs, dict)
        assert 0 <= total_conf <= 1

    def test_individual_confidences_keyed_by_measure_name(
        self, boolean_proportion_measure, mean_measure
    ):
        evaluation = Evaluation(
            measures=[boolean_proportion_measure, mean_measure],
            max_comparisons=1,
            sample_size=500,
        )
        _, confs = evaluation.compute_confidences()
        assert "accuracy" in confs
        assert "response_time" in confs

    def test_total_confidence_is_product(
        self, boolean_proportion_measure, mean_measure
    ):
        evaluation = Evaluation(
            measures=[boolean_proportion_measure, mean_measure],
            max_comparisons=1,
            sample_size=500,
        )
        total_conf, confs = evaluation.compute_confidences()
        expected_total = 1.0
        for conf in confs.values():
            expected_total *= conf
        assert math.isclose(total_conf, expected_total, rel_tol=1e-6)

    def test_adjusts_for_max_comparisons(self, boolean_proportion_measure):
        eval_1 = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=1,
            sample_size=500,
        )
        eval_3 = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=3,
            sample_size=500,
        )
        total_1, _ = eval_1.compute_confidences()
        total_3, _ = eval_3.compute_confidences()
        # More comparisons should reduce confidence for same sample size
        assert total_1 >= total_3

    def test_restores_measure_repetitions(self, measure_with_repetitions):
        original_reps = measure_with_repetitions.repetitions
        evaluation = Evaluation(
            measures=[measure_with_repetitions],
            max_comparisons=2,
            sample_size=500,
        )
        evaluation.compute_confidences()
        # Repetitions should be restored after computation
        assert measure_with_repetitions.repetitions == original_reps


class TestComputeAbsoluteErrors:
    """Tests for Evaluation.compute_absolute_errors."""

    def test_returns_errors_dict(self, evaluation_with_sample_size):
        # Need to set confidence for computing errors
        evaluation_with_sample_size.confidence = 0.95
        errors = evaluation_with_sample_size.compute_absolute_errors()
        assert isinstance(errors, dict)
        assert len(errors) == 1

    def test_errors_keyed_by_measure_name(
        self, boolean_proportion_measure, mean_measure
    ):
        evaluation = Evaluation(
            measures=[boolean_proportion_measure, mean_measure],
            max_comparisons=1,
            confidence=0.95,
            sample_size=500,
        )
        errors = evaluation.compute_absolute_errors()
        assert "accuracy" in errors
        assert "response_time" in errors

    def test_errors_are_positive(self, evaluation_with_sample_size):
        evaluation_with_sample_size.confidence = 0.95
        errors = evaluation_with_sample_size.compute_absolute_errors()
        for error in errors.values():
            assert error > 0

    def test_errors_decrease_with_sample_size(self, boolean_proportion_measure):
        eval_small = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=1,
            confidence=0.95,
            sample_size=100,
        )
        eval_large = Evaluation(
            measures=[boolean_proportion_measure],
            max_comparisons=1,
            confidence=0.95,
            sample_size=1000,
        )
        errors_small = eval_small.compute_absolute_errors()
        errors_large = eval_large.compute_absolute_errors()
        assert errors_small["accuracy"] > errors_large["accuracy"]

    def test_restores_measure_repetitions(self, measure_with_repetitions):
        original_reps = measure_with_repetitions.repetitions
        evaluation = Evaluation(
            measures=[measure_with_repetitions],
            max_comparisons=2,
            confidence=0.95,
            sample_size=500,
        )
        evaluation.compute_absolute_errors()
        # Repetitions should be restored after computation
        assert measure_with_repetitions.repetitions == original_reps


class TestEvaluationIntegration:
    """Integration tests for Evaluation workflows."""

    def test_compute_sample_size_then_confidences(self):
        """Verify that computed sample size achieves target confidence."""
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
            absolute_error=0.05,
        )
        evaluation = Evaluation(
            measures=[measure],
            max_comparisons=1,
            confidence=0.95,
        )
        sample_size = evaluation.compute_sample_size()
        evaluation.sample_size = sample_size
        total_conf, _ = evaluation.compute_confidences()
        # Achieved confidence should be at least the target
        assert total_conf >= 0.94  # Allow small numerical tolerance

    def test_compute_sample_size_then_errors(self):
        """Verify that computed sample size achieves target error."""
        target_error = 0.05
        measure = Measure(
            name="test",
            measure_type=MeasureType.PROPORTION_BOOLEAN,
            absolute_error=target_error,
        )
        evaluation = Evaluation(
            measures=[measure],
            max_comparisons=1,
            confidence=0.95,
        )
        sample_size = evaluation.compute_sample_size()
        evaluation.sample_size = sample_size
        errors = evaluation.compute_absolute_errors()
        # Achieved error should be at most the target
        assert errors["test"] <= target_error + 0.001  # Allow small numerical tolerance


class TestFilteredPopulation:
    """Tests for Evaluation with FilteredPopulation."""

    def test_filtered_population_get_size(self):
        """Test that FilteredPopulation computes conservative size estimate."""
        original = FinitePopulation(192)
        m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        eval1 = Evaluation([m1], original, confidence=0.97, sample_size=65)
        m1.empirical_value = 0.53
        filtered = original.filter_on(eval1, m1)
        # Conservative estimate: 192 * (0.53 + 0.1) = 192 * 0.63 = 121 (rounded up)
        expected_size = int(math.ceil(192 * (0.53 + 0.1)))
        assert filtered.get_size() == expected_size

    def test_filtered_population_from_infinite_is_infinite(self):
        """Test that filtering an infinite population results in infinite population."""
        infinite = InfinitePopulation()
        m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        eval1 = Evaluation([m1], infinite, confidence=0.95, sample_size=100)
        m1.empirical_value = 0.5
        filtered = infinite.filter_on(eval1, m1)
        assert filtered.is_infinite()
        assert filtered.get_size() == -1

    def test_evaluation_with_filtered_population_compute_confidences(
        self, filtered_population_evaluation
    ):
        """Test compute_confidences works with filtered population."""
        eval2, filtered, m1, m2 = filtered_population_evaluation
        total_conf, confs = eval2.compute_confidences()
        assert isinstance(total_conf, float)
        assert 0 <= total_conf <= 1
        assert "b" in confs

    def test_evaluation_with_filtered_population_compute_errors(
        self, filtered_population_evaluation
    ):
        """Test compute_absolute_errors works with filtered population."""
        eval2, filtered, m1, m2 = filtered_population_evaluation
        errors = eval2.compute_absolute_errors()
        assert "b" in errors
        assert errors["b"] > 0

    def test_evaluation_with_filtered_population_compute_sample_size(self):
        """Test compute_sample_size works with filtered population."""
        original = FinitePopulation(192)
        m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        eval1 = Evaluation([m1], original, confidence=0.97, sample_size=65)
        m1.empirical_value = 0.53
        filtered = original.filter_on(eval1, m1)

        m2 = Measure("b", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        eval2 = Evaluation([m2], filtered, confidence=0.96)
        sample_size = eval2.compute_sample_size()
        assert sample_size > 0
        assert isinstance(sample_size, int)

    def test_chained_filtered_population_size(self, chained_filtered_population):
        """Test that chained filtered populations compute correct size."""
        eval3, filtered_2, measures = chained_filtered_population
        m1, m2, m3 = measures
        # filtered_1 size: 192 * (0.53 + 0.1) = 121
        # filtered_2 size: 121 * (0.31 + 0.1) = 50 (rounded up)
        filtered_1_size = int(math.ceil(192 * (m1.empirical_value + m1.absolute_error)))
        expected_size = int(
            math.ceil(filtered_1_size * (m2.empirical_value + m2.absolute_error))
        )
        assert filtered_2.get_size() == expected_size

    def test_chained_filtered_population_evaluation(self, chained_filtered_population):
        """Test that evaluation on chained filtered population works."""
        eval3, filtered_2, measures = chained_filtered_population
        total_conf, confs = eval3.compute_confidences()
        assert isinstance(total_conf, float)
        assert 0 <= total_conf <= 1

    def test_filtered_population_with_categorical_measure(self):
        """Test FilteredPopulation with categorical measure (as in main.py)."""
        original = FinitePopulation(192)
        m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        eval1 = Evaluation([m1], original, confidence=0.97, sample_size=65)
        m1.empirical_value = 0.68
        filtered = original.filter_on(eval1, m1)

        m2 = Measure(
            "cat", MeasureType.PROPORTION_CATEGORICAL, absolute_error=0.1, categories=3
        )
        eval2 = Evaluation([m2], filtered, confidence=0.95, sample_size=24)
        total_conf, confs = eval2.compute_confidences()
        assert isinstance(total_conf, float)
        assert "cat" in confs

    def test_filtered_population_smaller_than_finite(self):
        """Test that filtered population is smaller than or equal to source."""
        original = FinitePopulation(1000)
        m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.05)
        eval1 = Evaluation([m1], original, confidence=0.95, sample_size=100)
        m1.empirical_value = 0.3
        filtered = original.filter_on(eval1, m1)
        assert filtered.get_size() <= original.get_size()

    def test_filter_on_multiple_measures(self):
        """Test filtering on multiple measures at once."""
        original = FinitePopulation(500)
        m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        m2 = Measure("b", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
        eval1 = Evaluation([m1, m2], original, confidence=0.95, sample_size=100)
        m1.empirical_value = 0.5
        m2.empirical_value = 0.4
        # Filter on both measures
        filtered = original.filter_on(eval1, [m1, m2])
        # Size should be: 500 * (0.5 + 0.1) * (0.4 + 0.1) = 500 * 0.6 * 0.5 = 150
        expected = int(
            math.ceil(
                500
                * (m1.empirical_value + m1.absolute_error)
                * (m2.empirical_value + m2.absolute_error)
            )
        )
        assert filtered.get_size() == expected
