import math
import pytest

from reeval.error_control import ErrorControl

from reeval.evaluation import (
    Evaluation,
    apply_cochran_finite_pop,
    compute_global_absolute_errors,
    compute_global_error_probabilities,
    compute_global_sample_sizes,
    reverse_cochran_finite_pop,
)
from reeval.measures.boolean_measure import BooleanMeasure
from reeval.measures.mean_measure import MeanMeasure
from reeval.measures.variance_measure import VarianceMeasure
from reeval.measures.rank_measure import RankMeasure
from reeval.population import FilteredPopulation, FinitePopulation, InfinitePopulation

POWER_ALPHA = 0.05


def ec_i(alpha: float = 0.05):
    return ErrorControl.type_i(alpha)


def ec_ii(beta: float = 0.05, alpha: float = POWER_ALPHA):
    return ErrorControl.type_ii(beta, significance_level=alpha)


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


def eval_inf(*measures, error_control=None):
    return Evaluation(
        measures=measures,
        population=InfinitePopulation(),
        error_control=ec_i() if error_control is None else error_control,
    )


def eval_fin(*measures, pop_size, error_control=None):
    return Evaluation(
        measures=measures,
        population=FinitePopulation(size=pop_size),
        error_control=ec_i() if error_control is None else error_control,
    )


def eval_filtered(
    *measures,
    source_population,
    filter_measure,
    empirical_proportion,
    filter_error_control,
    error_control=None,
):
    return Evaluation(
        measures=measures,
        population=FilteredPopulation(
            source_population=source_population,
            error_control=filter_error_control,
            filter_measure=filter_measure,
            empirical_proportion=empirical_proportion,
        ),
        error_control=ec_i() if error_control is None else error_control,
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
        n_tight = eval_inf(
            bool_measure(), error_control=ec_i(0.01)
        ).compute_sample_size()
        assert n_tight >= n_loose

    def test_type_ii_more_samples_than_type_i(self):
        """At the same nominal error level TYPE_II requires more samples because
        power calculations must satisfy both α and β constraints."""
        n_i = eval_inf(bool_measure(), error_control=ec_i()).compute_sample_size()
        n_ii = eval_inf(
            bool_measure(),
            error_control=ec_ii(),
        ).compute_sample_size()
        assert n_ii >= n_i

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
        n_loose = eval_fin(
            m, pop_size=5_000, error_control=ec_i(0.20)
        ).compute_sample_size()
        n_tight = eval_fin(
            m, pop_size=5_000, error_control=ec_i(0.01)
        ).compute_sample_size()
        assert n_tight >= n_loose


class TestComputeSampleSizeFiltered:
    def test_sample_size_matches_adjusted_downstream_problem(self):
        downstream_measure = bool_measure()
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.1)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=10_000),
            error_control=ec_i(0.01),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
        )
        evaluation = Evaluation(
            measures=(downstream_measure,),
            population=population,
            error_control=ec_i(0.05),
        )

        adjusted_error = population.adjust_error(ec_i(0.05))
        raw_required = Evaluation(
            measures=(downstream_measure,),
            population=InfinitePopulation(),
            error_control=adjusted_error,
        ).compute_sample_size()
        expected = apply_cochran_finite_pop(population.get_size(), raw_required)

        assert evaluation.compute_sample_size() == expected

    def test_default_error_probability_includes_filtering_guarantee(self):
        downstream_measure = bool_measure()
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.1)
        evaluation = eval_filtered(
            downstream_measure,
            source_population=FinitePopulation(size=10_000),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
            filter_error_control=ec_i(0.01),
            error_control=ec_i(0.05),
        )

        total_confidence, per_measure = evaluation.compute_error_probability()

        expected = evaluation.population.stage_success_probability()
        for confidence in per_measure.values():
            expected *= confidence
        assert total_confidence == pytest.approx(expected)

    def test_default_absolute_error_uses_adjusted_downstream_budget(self):
        downstream_measure = bool_measure()
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.1)
        evaluation = eval_filtered(
            downstream_measure,
            source_population=FinitePopulation(size=10_000),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
            filter_error_control=ec_i(0.01),
            error_control=ec_i(0.05),
        )

        sample_size = evaluation.compute_sample_size()
        errors = evaluation.compute_absolute_errors(sample_size=sample_size)

        raw_sample_size = reverse_cochran_finite_pop(
            evaluation.population.get_size(), sample_size
        )
        adjusted_error = evaluation.population.adjust_error(evaluation.error_control)
        expected_error = downstream_measure.compute_absolute_error(
            raw_sample_size,
            error_control=adjusted_error,
            repetition_multiplier=evaluation._get_total_repeats_(),
        )

        assert errors[downstream_measure.name] == pytest.approx(expected_error)


class TestComputeGlobalSampleSizes:
    def test_propagates_filtered_requirement_to_source_evaluation(self):
        source_eval = eval_fin(bool_measure(), pop_size=10_000, error_control=ec_i(0.2))
        child_population = FilteredPopulation(
            source_population=source_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.4,
        )
        child_eval = Evaluation(
            measures=(mean_measure(absolute_error=0.02),),
            population=child_population,
            error_control=ec_i(0.05),
        )

        sizes = compute_global_sample_sizes([source_eval, child_eval])

        assert sizes[child_eval] == child_eval.compute_sample_size()
        expected_source_requirement = int(
            math.ceil(
                sizes[child_eval] / child_population.conservative_lower_proportion()
            )
        )
        assert sizes[source_eval] >= expected_source_requirement

    def test_propagates_through_chained_filtered_populations(self):
        root_eval = eval_fin(bool_measure(), pop_size=20_000, error_control=ec_i(0.2))
        first_population = FilteredPopulation(
            source_population=root_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="first", absolute_error=0.05),
            empirical_proportion=0.5,
        )
        mid_eval = Evaluation(
            measures=(bool_measure(absolute_error=0.04),),
            population=first_population,
            error_control=ec_i(0.05),
        )
        second_population = FilteredPopulation(
            source_population=mid_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="second", absolute_error=0.05),
            empirical_proportion=0.7,
        )
        leaf_eval = Evaluation(
            measures=(mean_measure(absolute_error=0.05),),
            population=second_population,
            error_control=ec_i(0.05),
        )

        sizes = compute_global_sample_sizes([root_eval, mid_eval, leaf_eval])

        mid_requirement_from_leaf = int(
            math.ceil(
                sizes[leaf_eval] / second_population.conservative_lower_proportion()
            )
        )
        root_requirement_from_mid = int(
            math.ceil(
                sizes[mid_eval] / first_population.conservative_lower_proportion()
            )
        )

        assert sizes[mid_eval] >= mid_requirement_from_leaf
        assert sizes[root_eval] >= root_requirement_from_mid

    def test_raises_when_lower_filtered_proportion_can_be_zero(self):
        child_population = FilteredPopulation(
            source_population=FinitePopulation(size=10_000),
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.2),
            empirical_proportion=0.1,
        )
        child_eval = Evaluation(
            measures=(bool_measure(),),
            population=child_population,
            error_control=ec_i(0.05),
        )

        with pytest.raises(ValueError, match="may be empty"):
            compute_global_sample_sizes([child_eval])

    def test_raises_when_source_population_cannot_supply_enough_filtered_items(self):
        source_eval = eval_fin(bool_measure(), pop_size=100, error_control=ec_i(0.2))
        child_population = FilteredPopulation(
            source_population=source_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.1,
        )
        child_eval = Evaluation(
            measures=(mean_measure(absolute_error=0.01),),
            population=child_population,
            error_control=ec_i(0.05),
        )

        with pytest.raises(ValueError, match="too small"):
            compute_global_sample_sizes([source_eval, child_eval])


class TestGlobalEvaluationSummaries:
    def test_global_error_probabilities_use_provided_sample_sizes(self):
        source_eval = eval_fin(bool_measure(), pop_size=10_000, error_control=ec_i(0.2))
        child_population = FilteredPopulation(
            source_population=source_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.4,
        )
        child_eval = Evaluation(
            measures=(mean_measure(absolute_error=0.02),),
            population=child_population,
            error_control=ec_i(0.05),
        )

        sample_sizes = compute_global_sample_sizes([source_eval, child_eval])
        results = compute_global_error_probabilities(
            [source_eval, child_eval], sample_sizes
        )

        assert results[source_eval] == source_eval.compute_error_probability(
            sample_size=sample_sizes[source_eval]
        )
        assert results[child_eval] == child_eval.compute_error_probability(
            sample_size=sample_sizes[child_eval]
        )

    def test_global_absolute_errors_use_provided_sample_sizes(self):
        source_eval = eval_fin(bool_measure(), pop_size=10_000, error_control=ec_i(0.2))
        child_population = FilteredPopulation(
            source_population=source_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.4,
        )
        child_eval = Evaluation(
            measures=(mean_measure(absolute_error=0.02),),
            population=child_population,
            error_control=ec_i(0.05),
        )

        sample_sizes = compute_global_sample_sizes([source_eval, child_eval])
        results = compute_global_absolute_errors(
            [source_eval, child_eval], sample_sizes
        )

        assert results[source_eval] == source_eval.compute_absolute_errors(
            sample_size=sample_sizes[source_eval]
        )
        assert results[child_eval] == child_eval.compute_absolute_errors(
            sample_size=sample_sizes[child_eval]
        )

    def test_global_helpers_accept_per_evaluation_error_controls(self):
        source_eval = eval_fin(bool_measure(), pop_size=10_000, error_control=ec_i(0.2))
        child_population = FilteredPopulation(
            source_population=source_eval.population,
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.4,
        )
        child_eval = Evaluation(
            measures=(mean_measure(absolute_error=0.02),),
            population=child_population,
            error_control=ec_i(0.05),
        )
        overrides = {source_eval: ec_i(0.1), child_eval: ec_i(0.1)}
        sample_sizes = compute_global_sample_sizes([source_eval, child_eval])

        error_results = compute_global_error_probabilities(
            [source_eval, child_eval], sample_sizes, error_controls=overrides
        )
        absolute_results = compute_global_absolute_errors(
            [source_eval, child_eval], sample_sizes, error_controls=overrides
        )

        assert error_results[source_eval] == source_eval.compute_error_probability(
            sample_size=sample_sizes[source_eval],
            error_control=overrides[source_eval],
        )
        assert error_results[child_eval] == child_eval.compute_error_probability(
            sample_size=sample_sizes[child_eval],
            error_control=overrides[child_eval],
        )
        assert absolute_results[source_eval] == source_eval.compute_absolute_errors(
            sample_size=sample_sizes[source_eval],
            error_control=overrides[source_eval],
        )
        assert absolute_results[child_eval] == child_eval.compute_absolute_errors(
            sample_size=sample_sizes[child_eval],
            error_control=overrides[child_eval],
        )


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
        e = Evaluation(measures=[bool_measure()], error_control=ec_i(0.05))
        assert isinstance(e.population, InfinitePopulation)


# =========================================================================
# 5. Evaluation-level confidence properties
# =========================================================================


class TestComputeErrorProbability:
    def test_default_sample_size_meets_target_confidence(self):
        e = eval_inf(bool_measure(), mean_measure())
        confidence, per_measure = e.compute_error_probability()

        assert confidence >= 1 - e.error_control.alpha
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

    def test_type_ii_default_sample_size_meets_target_power(self):
        e = eval_inf(
            bool_measure(),
            mean_measure(),
            error_control=ec_ii(0.20),
        )
        power, per_measure = e.compute_error_probability()

        assert set(per_measure) == {m.name for m in e.measures}
        assert all(0 <= value <= 1 for value in per_measure.values())
        expected_power = 1
        for measure_power in per_measure.values():
            expected_power *= measure_power
        assert power == expected_power
        assert power >= 1 - e.error_control.beta

    def test_type_ii_uses_weakest_measure_power(self):
        e = eval_inf(
            bool_measure(),
            mean_measure(),
            error_control=ec_ii(0.20),
        )
        power, per_measure = e.compute_error_probability()

        expected_power = 1
        for measure_power in per_measure.values():
            expected_power *= measure_power
        assert power == expected_power

    def test_explicit_error_control_overrides_evaluation_default(self):
        e = eval_inf(bool_measure(), mean_measure(), error_control=ec_i())

        confidence, per_measure_confidence = e.compute_error_probability(
            error_control=ec_i()
        )
        power, per_measure_power = e.compute_error_probability(
            error_control=ec_ii(),
        )

        expected_confidence = 1
        for measure_confidence in per_measure_confidence.values():
            expected_confidence *= measure_confidence
        assert confidence == expected_confidence
        expected_power = 1
        for measure_power in per_measure_power.values():
            expected_power *= measure_power
        assert power == expected_power


class TestJointSampleSizeTarget:
    def test_multi_measure_type_i_sample_size_meets_joint_target(self):
        e = eval_inf(bool_measure(), mean_measure(), rank_measure())

        confidence, _ = e.compute_error_probability()

        assert confidence >= 1 - e.error_control.alpha

    def test_multi_measure_type_ii_sample_size_meets_joint_target(self):
        e = eval_inf(
            bool_measure(),
            mean_measure(),
            rank_measure(),
            error_control=ec_ii(0.20),
        )

        power, _ = e.compute_error_probability()

        assert power >= 1 - e.error_control.beta


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
