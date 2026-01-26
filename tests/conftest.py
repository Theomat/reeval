import pytest

from reeval.measure import Measure, MeasureType
from reeval.evaluation import Evaluation
from reeval.population import FinitePopulation


@pytest.fixture
def boolean_proportion_measure():
    """A basic boolean proportion measure."""
    return Measure(
        name="accuracy",
        measure_type=MeasureType.PROPORTION_BOOLEAN,
        absolute_error=0.05,
    )


@pytest.fixture
def categorical_proportion_measure():
    """A categorical proportion measure with multiple categories."""
    return Measure(
        name="category_distribution",
        measure_type=MeasureType.PROPORTION_CATEGORICAL,
        absolute_error=0.05,
        categories=5,
    )


@pytest.fixture
def mean_measure():
    """A mean measure with specified std."""
    return Measure(
        name="response_time",
        measure_type=MeasureType.MEAN,
        absolute_error=0.1,
        std=2.0,
    )


@pytest.fixture
def mean_measure_with_range():
    """A mean measure with value_range instead of std."""
    return Measure(
        name="score",
        measure_type=MeasureType.MEAN,
        absolute_error=0.5,
        value_range=(0, 100),
    )


@pytest.fixture
def variance_measure():
    """A variance measure."""
    return Measure(
        name="response_variance",
        measure_type=MeasureType.VARIANCE,
        absolute_error=0.1,
    )


@pytest.fixture
def measure_with_repetitions():
    """A measure with multiple repetitions."""
    return Measure(
        name="repeated_accuracy",
        measure_type=MeasureType.PROPORTION_BOOLEAN,
        absolute_error=0.05,
        repetitions=3,
    )


@pytest.fixture
def basic_evaluation(boolean_proportion_measure):
    """A basic evaluation with a single measure."""
    return Evaluation(
        measures=[boolean_proportion_measure],
        max_comparisons=1,
        confidence=0.95,
    )


@pytest.fixture
def multi_measure_evaluation(boolean_proportion_measure, mean_measure):
    """An evaluation with multiple measures."""
    return Evaluation(
        measures=[boolean_proportion_measure, mean_measure],
        max_comparisons=2,
        confidence=0.95,
    )


@pytest.fixture
def finite_population_evaluation(boolean_proportion_measure):
    """An evaluation with finite population."""
    return Evaluation(
        measures=[boolean_proportion_measure],
        max_comparisons=1,
        confidence=0.95,
        population=FinitePopulation(size=1000),
    )


@pytest.fixture
def evaluation_with_sample_size(boolean_proportion_measure):
    """An evaluation with sample_size specified."""
    return Evaluation(
        measures=[boolean_proportion_measure],
        max_comparisons=1,
        sample_size=500,
    )


@pytest.fixture
def filtered_population_evaluation():
    """An evaluation with a filtered population."""
    # Create the source population and first evaluation
    original = FinitePopulation(192)
    m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
    eval1 = Evaluation([m1], original, confidence=0.97, sample_size=65)
    m1.empirical_value = 0.53
    # Create filtered population
    filtered = original.filter_on(eval1, m1)
    # Create second evaluation on filtered population
    m2 = Measure("b", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
    eval2 = Evaluation([m2], filtered, confidence=0.96, sample_size=35)
    return eval2, filtered, m1, m2


@pytest.fixture
def chained_filtered_population():
    """A chain of filtered populations."""
    original = FinitePopulation(192)
    m1 = Measure("a", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
    eval1 = Evaluation([m1], original, confidence=0.97, sample_size=65)
    m1.empirical_value = 0.53
    filtered_1 = original.filter_on(eval1, m1)

    m2 = Measure("b", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
    eval2 = Evaluation([m2], filtered_1, confidence=0.96, sample_size=35)
    m2.empirical_value = 0.31
    filtered_2 = filtered_1.filter_on(eval2, m2)

    m3 = Measure("c", MeasureType.PROPORTION_BOOLEAN, absolute_error=0.1)
    eval3 = Evaluation([m3], filtered_2, confidence=0.95, sample_size=11)
    return eval3, filtered_2, [m1, m2, m3]
