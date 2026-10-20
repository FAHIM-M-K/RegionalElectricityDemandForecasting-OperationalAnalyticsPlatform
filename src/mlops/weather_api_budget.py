"""Small, testable request-attempt budget for weather-provider backfills."""


class ApiRequestBudgetExceeded(RuntimeError):
    """Raised before an HTTP request would exceed the configured hard cap."""


class ApiRequestBudget:
    """Count every outgoing request attempt, including retries and fallbacks."""

    def __init__(self, limit: int):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("API request budget must be a positive integer.")
        self.limit = limit
        self.used = 0

    def consume(self) -> int:
        """Reserve one request slot or stop before sending an over-budget call."""
        if self.used >= self.limit:
            raise ApiRequestBudgetExceeded(
                f"Weather API request budget exhausted ({self.used}/{self.limit} attempts)."
            )
        self.used += 1
        return self.used
