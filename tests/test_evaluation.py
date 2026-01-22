import math
import pytest

from reeval.measure import Measure, MeasureType
from reeval.evaluation import Evaluation


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
            population_size=float("inf"),
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
            population_size=pop_size,
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
            population_size=pop_size,
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
