from decimal import Decimal

import pytest

from evals.cost import estimate_deepseek_cost
from handshake.models.protocol import ModelUsage


@pytest.mark.parametrize(
    ("model", "peak", "expected"),
    [
        ("deepseek-flash", True, "0.000543"),
        ("deepseek-flash", False, "0.0002715"),
        ("deepseek-v4-flash", True, "0.000543"),
        ("deepseek-v4-pro", True, "0.002134"),
    ],
)
def test_estimate_charges_uncached_input_and_cache_reads_separately(
    model: str, peak: bool, expected: str
) -> None:
    usage = ModelUsage(input_tokens=1_000, output_tokens=200, cache_read_tokens=500)
    assert estimate_deepseek_cost(model, usage, peak=peak) == Decimal(expected)


@pytest.mark.parametrize(
    ("model", "usage"),
    [
        ("deepseek-flash", None),
        ("unknown", ModelUsage(10, 2)),
        ("deepseek-flash", ModelUsage(10, 2, cache_write_tokens=1)),
    ],
)
def test_estimate_leaves_unknown_cost_unavailable(model: str, usage: ModelUsage | None) -> None:
    assert estimate_deepseek_cost(model, usage, peak=True) is None
