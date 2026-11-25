"""Estimate DeepSeek text API cost from Handshake's recorded token counts."""

from decimal import Decimal

from handshake.models.protocol import ModelUsage

# DeepSeek pricing snapshot, 2026-11-25. Peak USD per million tokens, in the
# order uncached input, cache read, output. Off-peak rates are half of peak.
# https://api-docs.deepseek.com/quick_start/pricing/
PRICE_DATE = "2026-11-25"
_PEAK_RATES = {
    "deepseek-flash": (Decimal("0.30"), Decimal("0.006"), Decimal("1.20")),
    "deepseek-v4-pro": (Decimal("1.32"), Decimal("0.044"), Decimal("3.96")),
}
_FLASH_ALIASES = {"deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}


def estimate_deepseek_cost(model: str, usage: ModelUsage | None, *, peak: bool) -> Decimal | None:
    """Return estimated USD, or None when usage or an applicable rate is unavailable.

    ModelUsage input excludes cache reads and writes. DeepSeek's Anthropic
    endpoint currently reports zero cache writes and publishes no write rate.
    The caller selects the rate period for the supplied usage.
    """
    rates = _PEAK_RATES.get("deepseek-flash" if model in _FLASH_ALIASES else model)
    if usage is None or rates is None or usage.cache_write_tokens:
        return None

    input_rate, cache_read_rate, output_rate = rates
    cost = (
        usage.input_tokens * input_rate
        + usage.cache_read_tokens * cache_read_rate
        + usage.output_tokens * output_rate
    ) / 1_000_000
    return cost if peak else cost / 2
