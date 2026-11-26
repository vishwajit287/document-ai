
import os
from time import perf_counter


class UsageTracker:

    def __init__(self):

        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.execution_time = 0.0

        self.input_price = float(
            os.getenv("INPUT_PRICE_PER_1M", "0")
        )

        self.output_price = float(
            os.getenv("OUTPUT_PRICE_PER_1M", "0")
        )

    def add_usage(self, response):

        usage = getattr(
            response,
            "usage_metadata",
            None
        ) or {}

        self.requests += 1

        self.input_tokens += usage.get(
            "input_tokens", 0
        )

        self.output_tokens += usage.get(
            "output_tokens", 0
        )

        self.total_tokens += usage.get(
            "total_tokens", 0
        )

    def start_timer(self):
        return perf_counter()

    def stop_timer(self, start_time):
        self.execution_time += (
            perf_counter() - start_time
        )

    def get_cost(self):

        input_cost = (
            self.input_tokens
            * self.input_price
            / 1_000_000
        )

        output_cost = (
            self.output_tokens
            * self.output_price
            / 1_000_000
        )

        return round(input_cost + output_cost, 8)

    def get_metrics(self):

        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "execution_time_seconds": round(
                self.execution_time, 2
            ),
            "estimated_cost_usd": self.get_cost()
        }
