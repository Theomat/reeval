import pytest

from reeval.error_control import ErrorControl
from reeval.measures.boolean_measure import BooleanMeasure
from reeval.population import FilteredPopulation, FinitePopulation, InfinitePopulation


POWER_ALPHA = 0.05


def ec_i(alpha: float = 0.05):
    return ErrorControl.type_i(alpha)


def ec_ii(beta: float = 0.05, alpha: float = POWER_ALPHA):
    return ErrorControl.type_ii(beta, significance_level=alpha)


class TestFilteredPopulation:
    def test_filtered_population_size_uses_conservative_upper_bound(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.1)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_i(0.05),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
        )

        assert population.conservative_upper_proportion() == pytest.approx(0.3)
        assert population.get_size() == 301

    def test_filtered_population_size_is_capped_by_source_size(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.2)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_i(0.05),
            filter_measure=filter_measure,
            empirical_proportion=0.9,
        )

        assert population.conservative_upper_proportion() == pytest.approx(1.0)
        assert population.get_size() == 1000

    def test_filtered_population_lower_proportion_is_conservative(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.2)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_i(0.05),
            filter_measure=filter_measure,
            empirical_proportion=0.1,
        )

        assert population.conservative_lower_proportion() == pytest.approx(0.0)

    def test_filtered_infinite_population_remains_infinite(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.1)
        population = FilteredPopulation(
            source_population=InfinitePopulation(),
            error_control=ec_i(0.05),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
        )

        assert population.is_infinite()

    def test_adjust_error_preserves_joint_type_i_guarantee(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.05)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_i(0.01),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
        )

        adjusted = population.adjust_error(ec_i(0.05))

        assert adjusted.alpha == pytest.approx(1 - (1 - 0.05) / (1 - 0.01))

    def test_adjust_error_preserves_joint_power(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.05)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_ii(0.1),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
        )

        adjusted = population.adjust_error(ec_ii(0.2))

        assert adjusted.beta == pytest.approx(1 - (1 - 0.2) / (1 - 0.1))
        assert adjusted.significance_level == pytest.approx(POWER_ALPHA)

    def test_adjust_error_raises_when_target_is_unattainable(self):
        filter_measure = BooleanMeasure(name="keep", absolute_error=0.05)
        population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_i(0.2),
            filter_measure=filter_measure,
            empirical_proportion=0.2,
        )

        with pytest.raises(ValueError, match="unattainable"):
            population.adjust_error(ec_i(0.05))

    def test_stage_success_probability_matches_filter_error_control(self):
        alpha_population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_i(0.01),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.2,
        )
        power_population = FilteredPopulation(
            source_population=FinitePopulation(size=1000),
            error_control=ec_ii(0.2),
            filter_measure=BooleanMeasure(name="keep", absolute_error=0.05),
            empirical_proportion=0.2,
        )

        assert alpha_population.stage_success_probability() == pytest.approx(0.99)
        assert power_population.stage_success_probability() == pytest.approx(0.8)
