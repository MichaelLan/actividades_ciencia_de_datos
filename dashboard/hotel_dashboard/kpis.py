"""Business KPIs on filtered bookings and Spanish number formatting."""

from dataclasses import dataclass

import polars as pl


def fmt_number(value: float | None, decimals: int = 0) -> str:
    """Format with Spanish separators: 12.345,6."""
    if value is None:
        return "—"
    value = round(value, decimals) + 0.0
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_compact(value: float | None) -> str:
    """Compact Spanish format for large figures: 1,2 M / 845 mil."""
    if value is None:
        return "—"
    if abs(value) >= 1e6:
        return f"{fmt_number(value / 1e6, 1)} M"
    if abs(value) >= 1e4:
        return f"{fmt_number(value / 1e3, 0)} mil"
    return fmt_number(value)


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Format a share as a Spanish percentage: 37,3 %."""
    return "—" if value is None else f"{fmt_number(value * 100, decimals)} %"


@dataclass(frozen=True)
class BusinessKpis:
    """Headline indicators of a set of bookings."""

    bookings: int
    cancel_rate: float | None
    mean_adr: float | None
    revenue: float
    lost_revenue: float
    room_nights: int
    mean_lead_time: float | None
    mean_nights: float | None


def business_kpis(frame: pl.DataFrame) -> BusinessKpis:
    """Compute KPIs; ADR, revenue and room-nights use only kept bookings."""
    if frame.is_empty():
        return BusinessKpis(0, None, None, 0.0, 0.0, 0, None, None)
    stay_value = pl.col("adr") * pl.col("total_nights")
    kept = pl.col("is_canceled") == 0
    row = frame.select(
        pl.len().alias("bookings"),
        pl.col("is_canceled").mean().alias("cancel_rate"),
        pl.col("adr").filter(kept).mean().alias("mean_adr"),
        stay_value.filter(kept).sum().alias("revenue"),
        stay_value.filter(~kept).sum().alias("lost_revenue"),
        pl.col("total_nights").filter(kept).sum().alias("room_nights"),
        pl.col("lead_time").mean().alias("mean_lead_time"),
        pl.col("total_nights").mean().alias("mean_nights"),
    ).row(0, named=True)
    return BusinessKpis(
        bookings=int(row["bookings"]),
        cancel_rate=row["cancel_rate"],
        mean_adr=row["mean_adr"],
        revenue=float(row["revenue"] or 0.0),
        lost_revenue=float(row["lost_revenue"] or 0.0),
        room_nights=int(row["room_nights"] or 0),
        mean_lead_time=row["mean_lead_time"],
        mean_nights=row["mean_nights"],
    )


def monthly_kpis(frame: pl.DataFrame) -> pl.DataFrame:
    """Per arrival month KPI series (same definitions as ``business_kpis``) for sparklines."""
    stay_value = pl.col("adr") * pl.col("total_nights")
    kept = pl.col("is_canceled") == 0
    return (
        frame.group_by(pl.col("arrival_date").dt.truncate("1mo").alias("month"))
        .agg(
            pl.len().alias("bookings"),
            pl.col("is_canceled").mean().alias("cancel_rate"),
            pl.col("adr").filter(kept).mean().alias("mean_adr"),
            stay_value.filter(kept).sum().alias("revenue"),
            stay_value.filter(~kept).sum().alias("lost_revenue"),
            pl.col("lead_time").mean().alias("mean_lead_time"),
        )
        .sort("month")
        .fill_null(0)
    )
