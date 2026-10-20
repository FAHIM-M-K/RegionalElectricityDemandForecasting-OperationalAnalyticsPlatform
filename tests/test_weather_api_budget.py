import pytest

from src.mlops.weather_api_budget import ApiRequestBudget, ApiRequestBudgetExceeded


def test_budget_counts_each_request_attempt_and_stops_before_exceeding_limit():
    budget = ApiRequestBudget(3)

    assert [budget.consume(), budget.consume(), budget.consume()] == [1, 2, 3]
    with pytest.raises(ApiRequestBudgetExceeded, match="3/3 attempts"):
        budget.consume()
    assert budget.used == 3


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_budget_requires_positive_integer(limit):
    with pytest.raises(ValueError, match="positive integer"):
        ApiRequestBudget(limit)
