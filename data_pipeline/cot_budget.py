"""Provider-independent CNY-equivalent budget ledger for the CoT pilot."""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


def request_cost_cny(
    role: str, input_tokens: int, output_tokens: int, budget: dict[str, float]
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")
    if role == "teacher":
        return (
            input_tokens * float(budget["teacher_cny_per_million_input_tokens"])
            + output_tokens * float(budget["teacher_cny_per_million_output_tokens"])
        ) / 1_000_000
    if role == "validator":
        usd = (
            input_tokens * float(budget["validator_usd_per_million_input_tokens"])
            + output_tokens * float(budget["validator_usd_per_million_output_tokens"])
        ) / 1_000_000
        return usd * float(budget["usd_to_cny"])
    if role == "screener":
        return 0.0
    raise ValueError(f"unsupported budget role: {role}")


@dataclass
class BudgetLedger:
    hard_cap_cny: float
    stop_fraction: float
    spent_cny: float = 0.0

    @property
    def stop_limit_cny(self) -> float:
        return self.hard_cap_cny * self.stop_fraction

    def assert_can_spend(self, estimated_cny: float) -> None:
        if estimated_cny < 0:
            raise ValueError("estimated cost cannot be negative")
        if self.spent_cny + estimated_cny > self.stop_limit_cny:
            raise BudgetExceeded(
                f"budget stop: {self.spent_cny + estimated_cny:.6f} CNY would "
                f"exceed {self.stop_limit_cny:.6f} CNY"
            )

    def record(self, actual_cny: float) -> None:
        if actual_cny < 0:
            raise ValueError("actual cost cannot be negative")
        self.spent_cny += actual_cny
