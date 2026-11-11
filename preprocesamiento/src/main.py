"""Run from preprocesamiento: uv run python -m src.main --help."""

import argparse
from pathlib import Path

from src.cleaning import clean_bookings
from src.data import prepare, write_json
from src.eda import explore
from src.modeling import SEED, run_task

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible Hotel Booking Demand phases 1–2. Raw data are preserved; models are retrospective, not booking-time forecasts."
    )
    parser.add_argument(
        "command",
        choices=["prepare", "eda", "phase2", "all"],
        help="prepare downloads/validates; eda explores the original; phase2 cleans/splits/fits basic baselines; all runs every stage",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data/raw",
        help="Original CSV directory (download skipped when CSV exists)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/processed",
        help="Task/split Parquet directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Readable JSON diagnostics directory",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT.parent / "informe/imagenes",
        help="Independent vector SVG destination",
    )
    args = parser.parse_args()
    frame = prepare(args.raw_dir, args.output_dir)
    print(f"Validated original dataset: {frame.height} rows, {frame.width} columns")
    if args.command in {"eda", "all"}:
        explore(frame, args.output_dir, args.figures_dir)
        print(
            f"EDA written to {args.output_dir / 'eda.json'}; SVG figures: {args.figures_dir}"
        )
    if args.command in {"phase2", "all"}:
        cleaned, audit = clean_bookings(frame)
        args.processed_dir.mkdir(parents=True, exist_ok=True)
        cleaned.write_parquet(args.processed_dir / "cleaned.parquet")
        write_json(args.output_dir / "cleaning.json", audit)
        results = {
            task: run_task(cleaned, task, args.processed_dir)
            for task in ["classification", "regression"]
        }
        write_json(
            args.output_dir / "phase2.json",
            {
                "seed": SEED,
                "tasks": results,
                "interpretation": "Fixed baselines, no hyperparameter selection; validation and test reported separately. Snapshot is day prior to arrival, not creation. Arrival-based random grouped holdouts do not establish future performance. Regression ADR=sum lodging transactions/nights in non-canceled bookings; conditional population has selection bias. Finite extreme ADR retained; metrics are tail-sensitive. Hash collision risk is negligible, not mathematically impossible.",
                "excluded_features": {
                    "reservation_status": "outcome",
                    "reservation_status_date": "outcome date",
                    "adr": "transaction-derived outcome, not proven quoted tariff",
                    "assigned_room_type": "operational assignment",
                    "booking_changes": "cumulative until check-in/cancellation",
                    "days_in_waiting_list": "post-confirmation information",
                    "deposit_type": "payments before arrival OR cancellation",
                    "required_car_parking_spaces": "unknown individual request timestamp",
                    "total_of_special_requests": "cumulative requests",
                    "country": "may be known only at check-in",
                    "agent/company": "nominal identifiers; NULL means not applicable",
                    "arrival_date_year/week/day": "omit redundant calendar components; month is nominal",
                    "is_canceled": "classification target; regression population condition only",
                },
            },
        )
        print(
            f"Phase 2: {cleaned.height} retained rows; results: {args.output_dir / 'phase2.json'}"
        )


if __name__ == "__main__":
    main()
