"""Prepare Typst data and report-only figures without changing pipeline outputs."""

import hashlib
import json
from pathlib import Path


def render_figures(root: Path, eda: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    frame = pl.read_csv(
        root / "preprocesamiento/data/raw/hotel_bookings.csv",
        null_values=["NULL", "NA", ""],
        infer_schema_length=10000,
    )
    directory = root / "informe/imagenes"
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    colors = ("#246b82", "#c76b44")

    def save(fig, name):
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7), layout="constrained")
    for ax, name, limit, label in zip(
        axes[:2],
        ("adr", "lead_time"),
        (300, 500),
        ("ADR", "Anticipación (días)"),
        strict=True,
    ):
        values = frame[name].drop_nulls().to_numpy()
        ax.hist(values, bins=np.linspace(0, limit, 41), color=colors[0])
        ax.set(xlabel=label, ylabel="Reservas", xlim=(0, limit))
    hotels = ("City Hotel", "Resort Hotel")
    axes[2].boxplot(
        [frame.filter(pl.col("hotel") == h)["adr"].to_numpy() for h in hotels],
        tick_labels=("City", "Resort"),
        showfliers=False,
    )
    axes[2].set(xlabel="Hotel", ylabel="ADR")
    save(fig, "distributions")

    months = {
        m: i + 1
        for i, m in enumerate(
            (
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            )
        )
    }
    dated = frame.with_columns(
        pl.date(
            pl.col("arrival_date_year"),
            pl.col("arrival_date_month").replace_strict(months),
            pl.col("arrival_date_day_of_month"),
        ).alias("arrival")
    )
    monthly = (
        dated.group_by(pl.col("arrival").dt.truncate("1mo"), "hotel")
        .len()
        .sort("arrival")
    )
    fig, ax = plt.subplots(figsize=(3.7, 2.1), layout="constrained")
    for h, color in zip(hotels, colors, strict=True):
        group = monthly.filter(pl.col("hotel") == h)
        ax.plot(group["arrival"].to_list(), group["len"], label=h, color=color)
    ax.legend(fontsize=8, frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set(xlabel="Mes de llegada", ylabel="Reservas")
    save(fig, "arrivals")

    missing = {
        k: v["percent"] for k, v in eda["quality"]["missing"].items() if v["count"]
    }
    fig, ax = plt.subplots(figsize=(3.7, 1.6), layout="constrained")
    ax.barh(list(missing), list(missing.values()), color=colors[0])
    ax.set(xlabel="Ausencias (%)", xlim=(0, 100))
    save(fig, "missingness")

    selected = (
        "lead_time",
        "stays_in_week_nights",
        "stays_in_weekend_nights",
        "adults",
        "children",
        "total_of_special_requests",
        "adr",
        "is_canceled",
    )
    labels = (
        "Anticipación",
        "Noches labor.",
        "Noches fin sem.",
        "Adultos",
        "Niños",
        "Solicitudes",
        "ADR",
        "Cancelación",
    )
    indices = [eda["correlations"]["columns"].index(k) for k in selected]
    matrix = np.array(eda["correlations"]["spearman"])[np.ix_(indices, indices)]
    fig, ax = plt.subplots(figsize=(4.3, 3.8), layout="constrained")
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(labels)):
        for j in range(i):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if abs(matrix[i, j]) > 0.6 else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.75, ticks=(-1, 0, 1))
    save(fig, "correlations")

    sample = frame.sample(n=4000, seed=42).with_columns(
        (pl.col("stays_in_week_nights") + pl.col("stays_in_weekend_nights")).alias(
            "total_nights"
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), layout="constrained")
    for ax, name, limit, label in zip(
        axes,
        ("lead_time", "total_nights"),
        (500, 21),
        ("Anticipación (días)", "Duración total (noches)"),
        strict=True,
    ):
        for status, color, legend in zip(
            (0, 1), colors, ("No cancelada", "Cancelada"), strict=True
        ):
            group = sample.filter(pl.col("is_canceled") == status)
            ax.scatter(
                group[name], group["adr"], s=5, alpha=0.25, color=color, label=legend
            )
        ax.set(xlabel=label, ylabel="ADR", xlim=(0, limit), ylim=(0, 300))
    axes[0].legend(fontsize=8, frameon=False, markerscale=2)
    save(fig, "relationships")

    fig = plt.figure(figsize=(8, 3.6), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[1, 0]),
    )
    for ax, name, title in zip(
        axes,
        ("hotel", "market_segment", "deposit_type"),
        ("Hotel", "Segmento", "Depósito"),
        strict=True,
    ):
        table = (
            frame.group_by(name)
            .agg(pl.col("is_canceled").mean().alias("rate"), pl.len().alias("n"))
            .sort("rate")
        )
        labels = [
            f"{str(r[name]).replace(' Hotel', '')} ({r['n']})" for r in table.to_dicts()
        ]
        ax.barh(labels, table["rate"] * 100, color=colors[0])
        ax.set(title=title, xlabel="Cancelación (%)", xlim=(0, 100))
        ax.tick_params(axis="y", labelsize=9)
    save(fig, "cancellation_categories")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    outputs = root / "preprocesamiento" / "outputs"
    payload = {}
    hashes = {}
    for name in ("eda", "source"):
        content = (outputs / f"{name}.json").read_bytes()
        payload[name] = json.loads(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    payload["input_sha256"] = hashes
    # Solo transporta datos; el redondeo pertenece a la presentación en Typst.
    literal = json.dumps(json.dumps(payload, ensure_ascii=False), ensure_ascii=False)
    destination = root / "informe" / "datos.typ"
    destination.write_text(
        "// Generated by generar_datos.py from existing JSON; do not edit.\n"
        f"#let results = json(bytes({literal}))\n",
        encoding="utf-8",
    )
    print(f"Serialized eda.json and source.json to {destination.relative_to(root)}")
    for name, digest in hashes.items():
        print(f"{name}.json SHA-256: {digest}")
    render_figures(root, payload["eda"])


if __name__ == "__main__":
    main()
