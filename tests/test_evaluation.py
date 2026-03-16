import pytest

from reeval.error_type import ErrorType
from reeval.evaluation import (
    Evaluation,
    apply_cochran_finite_pop,
    reverse_cochran_finite_pop,
)
from reeval.measures.boolean_measure import BooleanMeasure
from reeval.measures.mean_measure import MeanMeasure
from reeval.measures.variance_measure import VarianceMeasure
from reeval.measures.rank_measure import RankMeasure
from reeval.population import FinitePopulation, InfinitePopulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bool_measure(absolute_error=0.05):
    return BooleanMeasure(name="b", absolute_error=absolute_error)


def mean_measure(std=1.0, absolute_error=0.1):
    return MeanMeasure(name="m", std=std, absolute_error=absolute_error)


def var_measure(relative_error=0.1):
    return VarianceMeasure(name="v", relative_error=relative_error)


def rank_measure(max_rank=10, absolute_error=0.5):
    return RankMeasure(name="r", max_rank=max_rank, absolute_error=absolute_error)


def eval_inf(*measures, error=0.05, error_type=ErrorType.TYPE_I):
    return Evaluation(
        measures=measures,
        population=InfinitePopulation(),
        error_control=(error, error_type),
    )


def eval_fin(*measures, pop_size, error=0.05, error_type=ErrorType.TYPE_I):
    return Evaluation(
        measures=measures,
        population=FinitePopulation(size=pop_size),
        error_control=(error, error_type),
    )


# =========================================================================
# 1. Cochran helpers – unit properties
# =========================================================================


class TestCochranHelpers:
    def test_apply_reduces_sample_size(self):
        """Cochran adjustment should never increase the sample size."""
        n0 = 500
        n = apply_cochran_finite_pop(pop_size=1000, n0=n0)
        assert n <= n0

    def test_apply_returns_positive(self):
        n = apply_cochran_finite_pop(pop_size=200, n0=100)
        assert n > 0

    def test_apply_increases_with_population_size(self):
        """Larger population -> adjustment is less aggressive -> larger corrected n."""
        n_small = apply_cochran_finite_pop(pop_size=100, n0=80)
        n_large = apply_cochran_finite_pop(pop_size=10_000, n0=80)
        assert n_large >= n_small

    def test_apply_approaches_n0_for_very_large_population(self):
        """For an enormous population the correction should be negligible."""
        n0 = 400
        n = apply_cochran_finite_pop(pop_size=10_000_000, n0=n0)
        assert abs(n - n0) <= 1  # at most one rounding unit off

    def test_reverse_is_approximate_inverse_of_apply(self):
        """round-trip: reverse(apply(pop, n0)) ≈ n0."""
        pop_size = 5_000
        for n0 in [50, 200, 500, 1000]:
            n = apply_cochran_finite_pop(pop_size, n0)
            recovered = reverse_cochran_finite_pop(pop_size, n)
            assert (
                abs(recovered - n0) <= 2
            ), f"round-trip failed: n0={n0}, n={n}, recovered={recovered}"

    def test_apply_result_less_than_population_size(self):
        """You cannot sample more items than the population contains."""
        pop_size = 300
        n = apply_cochran_finite_pop(pop_size=pop_size, n0=500)
        assert n <= pop_size


# =========================================================================
# 2. compute_sample_size – basic properties (InfinitePopulation)
# =========================================================================


class TestComputeSampleSizeInfinite:
    @pytest.mark.parametrize(
        "measures",
        [
            [bool_measure()],
            [mean_measure()],
            [var_measure()],
            [rank_measure()],
            [bool_measure(), mean_measure()],
        ],
    )
    def test_returns_positive_integer(self, measures):
        n = eval_inf(*measures).compute_sample_size()
        assert isinstance(n, (int, float))
        assert n > 0

    def test_monotone_decreasing_with_error(self):
        """Lower error tolerance requires more samples."""
        n_loose = eval_inf(bool_measure()).compute_sample_size()  # default error=0.05
        n_tight = eval_inf(bool_measure(), error=0.01).compute_sample_size()
        assert n_tight >= n_loose

    def test_type_ii_fewer_samples_than_type_i(self):
        """At the same error level TYPE_II requires fewer (or equal) samples than TYPE_I."""
        n_i = eval_inf(
            bool_measure(), error_type=ErrorType.TYPE_I
        ).compute_sample_size()
        n_ii = eval_inf(
            bool_measure(), error_type=ErrorType.TYPE_II
        ).compute_sample_size()
        assert n_ii <= n_i

    def test_dominated_by_most_demanding_measure(self):
        """The sample size of an evaluation is at least that of each constituent measure."""
        m_tight = mean_measure(std=1.0, absolute_error=0.01)  # very demanding
        m_loose = bool_measure(absolute_error=0.40)  # very easy

        n_tight_alone = eval_inf(m_tight).compute_sample_size()
        n_joint = eval_inf(m_tight, m_loose).compute_sample_size()
        # Bonferroni correction from the extra measure may push n_joint above n_tight_alone,
        # but it should never be below the dominant measure's standalone sample size.
        assert n_joint >= n_tight_alone

    def test_adding_measure_does_not_decrease_sample_size(self):
        """Adding another measure can only maintain or increase the required sample size."""
        m1 = bool_measure()
        m2 = mean_measure()
        n_single = eval_inf(m1).compute_sample_size()
        n_two = eval_inf(m1, m2).compute_sample_size()
        assert n_two >= n_single

    def test_bonferroni_correction_increases_with_repeats(self):
        """More total repeats (Bonferroni) must not decrease sample size."""
        m_few = BooleanMeasure(name="b", repeats=1, absolute_error=0.05)
        m_many = BooleanMeasure(name="b", repeats=10, absolute_error=0.05)
        n_few = eval_inf(m_few).compute_sample_size()
        n_many = eval_inf(m_many).compute_sample_size()
        assert n_many >= n_few

    def test_tighter_tolerance_needs_more_samples_across_measure_types(self):
        pairs = [
            (bool_measure(absolute_error=0.10), bool_measure(absolute_error=0.01)),
            (mean_measure(absolute_error=0.20), mean_measure(absolute_error=0.02)),
            (var_measure(relative_error=0.20), var_measure(relative_error=0.02)),
            (rank_measure(absolute_error=1.0), rank_measure(absolute_error=0.1)),
        ]
        for m_loose, m_tight in pairs:
            n_loose = eval_inf(m_loose).compute_sample_size()
            n_tight = eval_inf(m_tight).compute_sample_size()
            assert (
                n_tight > n_loose
            ), f"{type(m_loose).__name__}: tight should need more samples"


# =========================================================================
# 3. compute_sample_size – FinitePopulation properties
# =========================================================================


class TestComputeSampleSizeFinite:
    def test_finite_less_than_or_equal_to_infinite(self):
        """Cochran's formula always gives a smaller or equal sample size than infinite."""
        m = bool_measure()
        n_inf = eval_inf(m).compute_sample_size()
        n_fin = eval_fin(m, pop_size=10_000).compute_sample_size()
        assert n_fin <= n_inf

    def test_larger_population_needs_more_samples(self):
        """A larger population requires a larger corrected sample size (approaches infinite)."""
        m = bool_measure()
        n_small_pop = eval_fin(m, pop_size=200).compute_sample_size()
        n_large_pop = eval_fin(m, pop_size=50_000).compute_sample_size()
        assert n_large_pop >= n_small_pop

    def test_very_large_population_approaches_infinite(self):
        """With a very large population, finite ≈ infinite sample size."""
        m = bool_measure()
        n_inf = eval_inf(m).compute_sample_size()
        n_fin = eval_fin(m, pop_size=10_000_000).compute_sample_size()
        assert abs(n_fin - n_inf) <= 1

    def test_finite_sample_at_most_population_size(self):
        """The corrected sample size should not exceed the population."""
        m = bool_measure(absolute_error=0.001)  # requires many samples
        pop_size = 500
        n = eval_fin(m, pop_size=pop_size).compute_sample_size()
        assert n <= pop_size

    def test_finite_monotone_decreasing_with_error(self):
        """Lower error requires more samples even for finite populations."""
        m = bool_measure()
        n_loose = eval_fin(m, pop_size=5_000, error=0.20).compute_sample_size()
        n_tight = eval_fin(m, pop_size=5_000, error=0.01).compute_sample_size()
        assert n_tight >= n_loose


# =========================================================================
# 4. Evaluation structure properties
# =========================================================================


class TestEvaluationStructure:
    def test_measures_stored_as_tuple(self):
        e = eval_inf(bool_measure(), mean_measure())
        assert isinstance(e.measures, tuple)

    def test_measures_count_preserved(self):
        measures = [bool_measure(), mean_measure(), var_measure()]
        e = eval_inf(*measures)
        assert len(e.measures) == len(measures)

    def test_total_repeats_is_sum_of_individual_repeats(self):
        m1 = BooleanMeasure(name="b1", repeats=3, absolute_error=0.05)
        m2 = BooleanMeasure(name="b2", repeats=5, absolute_error=0.05)
        e = eval_inf(m1, m2)
        assert e._get_total_repeats_() == 8

    def test_single_measure_total_repeats(self):
        m = BooleanMeasure(name="b", repeats=4, absolute_error=0.05)
        e = eval_inf(m)
        assert e._get_total_repeats_() == 4

    def test_default_population_is_infinite(self):
        e = Evaluation(
            measures=[bool_measure()], error_control=(0.05, ErrorType.TYPE_I)
        )
        assert isinstance(e.population, InfinitePopulation)


# =========================================================================
# 5. Evaluation-level confidence properties
# =========================================================================


class TestComputeErrorProbability:
    def test_default_sample_size_meets_target_confidence(self):
        e = eval_inf(bool_measure(), mean_measure())
        confidence, per_measure = e.compute_error_probability()

        assert confidence >= 1 - e.error_control[0]
        assert set(per_measure) == {m.name for m in e.measures}
        assert all(0 <= value <= 1 for value in per_measure.values())

    def test_increases_with_sample_size(self):
        e = eval_inf(bool_measure(), mean_measure())
        n_base = e.compute_sample_size()

        confidence_small, per_measure_small = e.compute_error_probability(
            sample_size=max(1, n_base // 2)
        )
        confidence_large, per_measure_large = e.compute_error_probability(
            sample_size=n_base * 2
        )

        assert confidence_large >= confidence_small
        for name in per_measure_small:
            assert per_measure_large[name] >= per_measure_small[name]

    def test_finite_population_confidence_increases_with_sample_size(self):
        e = eval_fin(bool_measure(), mean_measure(), pop_size=50_000)
        n_base = e.compute_sample_size()

        confidence_small, _ = e.compute_error_probability(
            sample_size=max(1, n_base // 2)
        )
        confidence_large, _ = e.compute_error_probability(
            sample_size=min(2 * n_base, 40_000)
        )

        assert confidence_large >= confidence_small


# =========================================================================
# 6. Evaluation-level error properties
# =========================================================================


class TestComputeAbsoluteErrors:
    def test_decreases_with_sample_size(self):
        measures = [bool_measure(), mean_measure(), rank_measure()]
        e = eval_inf(*measures)
        n_base = e.compute_sample_size()

        errors_small = e.compute_absolute_errors(sample_size=max(1, n_base // 2))
        errors_large = e.compute_absolute_errors(sample_size=n_base * 2)

        for measure in measures:
            assert errors_large[measure.name] <= errors_small[measure.name]

    def test_default_sample_size_respects_requested_tolerances(self):
        measures = [bool_measure(), mean_measure(), rank_measure()]
        e = eval_inf(*measures)

        errors = e.compute_absolute_errors()

        for measure in measures:
            assert errors[measure.name] <= measure.absolute_error
