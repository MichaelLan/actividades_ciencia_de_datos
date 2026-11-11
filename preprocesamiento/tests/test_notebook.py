"""Contrato offline del cuaderno, sin dependencias opcionales de Jupyter."""

import ast
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "fase2.ipynb"


def notebook_code():
    """Lee las celdas como JSON estándar, sin iniciar un kernel."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.fixture
def definitions():
    """Extrae funciones/imports y constantes puras; no ejecuta datos ni exportaciones."""
    _, code = notebook_code()
    tree = ast.parse(code)
    constants = {"COLUMNS", "CATEGORICAL", "STRINGS", "SCHEMA", "DATASET", "SEED"}
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        )
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace


def bookings(size=480):
    """Construye reservas pequeñas y distintas sin descargar ni leer el CSV real."""
    return pl.DataFrame(
        {
            "original_fingerprint": list(range(size)),
            "is_canceled": [i % 2 for i in range(size)],
        }
    )


def test_notebook_is_standalone_and_documented():
    notebook, code = notebook_code()
    assert notebook["nbformat"] == 4
    tree = ast.parse(code)
    assert len(notebook["cells"]) >= 20
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("src")
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("src") for alias in node.names)
        if isinstance(node, ast.FunctionDef):
            assert ast.get_docstring(node), node.name
    assert "uv run" not in code
    assert "assert sys.version_info[:2] == (3, 14)" in code
    assert '"outputs/notebook_phase2"' in code
    assert '"data/processed/notebook_phase2"' in code


@pytest.mark.parametrize("classification", [False, True])
def test_notebook_weighted_relative_deficit(definitions, classification):
    frame = bookings()
    weighted = pl.concat([frame, *[frame.head(48)] * 3])
    split = definitions["split_bookings"]
    parts = split(weighted, classification)
    assert [part.height for part in parts.values()] == [468, 78, 78]
    assert sum(part.height for part in parts.values()) == weighted.height
    seen = set()
    for name, part in parts.items():
        fingerprints = set(part["original_fingerprint"])
        assert not seen & fingerprints
        seen |= fingerprints
        sizes = part.group_by("original_fingerprint").len()
        repeated = sizes.filter(pl.col("len") > 1)["len"].sum()
        assert repeated > 0
        assert abs(repeated / part.height - 192 / 624) < 0.04
        assert part.equals(split(weighted, classification)[name])
        if classification:
            assert abs(part["is_canceled"].mean() - 0.5) < 0.04
    assert seen == set(weighted["original_fingerprint"])


def test_notebook_rejects_impossible_before_export(definitions, tmp_path):
    frame = bookings(400)
    weighted = pl.concat([frame, *[frame.head(1)] * 4000])
    with pytest.raises(ValueError, match="Cannot satisfy.*70.*80.*4001.*4400"):
        definitions["run_task"](weighted, "classification", tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_notebook_adults_and_train_only_transforms(definitions, task):
    numeric, categorical = definitions["feature_columns"](task)
    assert "adults" in numeric and "total_guests" not in numeric + categorical
    assert not set(numeric + categorical) & {
        "adr",
        "is_canceled",
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "booking_changes",
        "deposit_type",
        "country",
        "agent",
        "company",
        "days_in_waiting_list",
        "total_of_special_requests",
        "required_car_parking_spaces",
    }
    train = pl.DataFrame(
        {"adults": [2, 2, 2, 2], "children": [0.0, 1.0, 1.0, 2.0], "hotel": ["A"] * 4}
    )
    processor = definitions["make_preprocessor"](["adults", "children"], ["hotel"])
    processor.fit(train)
    branch = processor.named_transformers_["numeric"]
    for adults in [3, 5]:
        holdout = pl.DataFrame(
            {"adults": [adults], "children": [None]},
            schema_overrides={"children": pl.Float64},
        ).with_columns(pl.lit("NEW").alias("hotel"))
        values = branch.named_steps["imputer"].transform(
            holdout.select("adults", "children")
        )
        np.testing.assert_array_equal(values, [[adults, 1]])
        assert np.isfinite(processor.transform(holdout)).all()
    np.testing.assert_array_equal(branch.named_steps["imputer"].statistics_, [2, 1])
    np.testing.assert_array_equal(branch.named_steps["scale"].mean_, [2, 1])
    assert processor.named_transformers_["categorical"].named_steps[
        "encode"
    ].categories_[0].tolist() == ["A"]


def test_notebook_schema_and_cleaning_preserve_raw(definitions, tmp_path):
    from test_pipeline import fixture_frame

    path = tmp_path / "real_schema_fixture.csv"
    raw = fixture_frame(6).with_columns(
        pl.Series("adr", [80.0, -1.0, 0.0, 5400.0, 20.0, 30.0]),
        pl.Series("stays_in_week_nights", [2, 2, 2, 2, 0, 2]),
    )
    raw = pl.concat([raw, raw.head(1)])
    raw.write_csv(path)
    before = path.read_bytes()
    loaded = definitions["read_bookings"](path)
    assert loaded.schema["agent"] == loaded.schema["company"] == pl.String
    assert loaded.schema["children"] == pl.Float64
    cleaned, audit = definitions["clean_bookings"](loaded)
    assert audit["removed_rows"] == 2
    assert cleaned.height == 5
    assert cleaned["total_guests"].null_count() == 5
    assert cleaned["total_nights"].to_list() == [2] * 5
    assert cleaned["original_fingerprint"].n_unique() == 4
    assert 0.0 in cleaned["adr"] and 5400.0 in cleaned["adr"]
    assert path.read_bytes() == before
    pl.DataFrame({"bad": [1]}).write_csv(path)
    with pytest.raises(ValueError, match="Expected 32 columns"):
        definitions["read_bookings"](path)


def test_notebook_null_tokens_match_original_loader(definitions, tmp_path):
    from test_pipeline import fixture_frame

    path = tmp_path / "tokens.csv"
    raw = fixture_frame(3).with_columns(
        pl.Series("children", ["NA", "NULL", ""]),
        pl.Series("agent", ["NULL", "", "NA"]),
        pl.Series("company", ["", "NA", "NULL"]),
    )
    raw.write_csv(path)
    before = path.read_bytes()
    frame = definitions["read_bookings"](path)
    for name in ["agent", "company", "children"]:
        assert frame[name].null_count() == 3
    assert path.read_bytes() == before


def test_notebook_600_rows_and_exact_metrics(definitions):
    frame = bookings(400)
    weighted = pl.concat([frame, *[frame.head(1)] * 200])
    parts = definitions["split_bookings"](weighted, classification=True)
    assert [part.height for part in parts.values()] == [450, 75, 75]
    assert all(
        abs(part["is_canceled"].mean() - 1 / 3) < 0.015 for part in parts.values()
    )
    scores = definitions["score_predictions"](
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0.1, 0.9, 0.4, 0.2])
    )
    assert scores == {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "confusion_matrix": [[2, 0], [1, 1]],
        "roc_auc": 1.0,
    }


@pytest.mark.parametrize(
    ("positive_weights", "message"),
    [
        ([60], "class 1 has 1 original groups; at least 3 required"),
        ([30, 30], "class 1 has 2 original groups; at least 3 required"),
        (
            [20, 20, 20],
            "Weighted greedy assignment did not preserve all target classes",
        ),
        ([], "Classification requires both target classes [0, 1]"),
        ([600], "Classification requires both target classes [0, 1]"),
    ],
    ids=[
        "F1-600-one-minority-group",
        "two-minority-groups",
        "three-groups-invalid-greedy",
        "no-canceled",
        "no-uncanceled",
    ],
)
def test_notebook_rejects_unrepresented_classes_before_fit_or_export(
    definitions, tmp_path, monkeypatch, positive_weights, message
):
    """Rechaza por interfaz completa sin ajustar modelos ni exportar particiones."""
    base: dict[str, str | int | float] = {
        name: "A" if dtype == pl.String else 0
        for name, dtype in definitions["SCHEMA"].items()
    }
    base.update(
        adults=2,
        children=0.0,
        babies=0,
        stays_in_week_nights=2,
        stays_in_weekend_nights=0,
        adr=100.0,
        is_canceled=1,
    )
    rows = [
        base | {"lead_time": index}
        for index, weight in enumerate(positive_weights)
        for _ in range(weight)
    ] + [
        base | {"lead_time": index + 1000, "is_canceled": 0}
        for index in range(600 - sum(positive_weights))
    ]
    raw = pl.DataFrame(rows, schema_overrides=definitions["SCHEMA"])
    original = raw.clone()
    frame, audit = definitions["clean_bookings"](raw)
    assert frame.height == 600 and audit["removed_rows"] == 0
    assert frame["is_canceled"].sum() == sum(positive_weights)
    assert frame["original_fingerprint"].n_unique() == 600 - sum(
        positive_weights
    ) + len(positive_weights)
    destination = tmp_path / "rejected"
    fit_calls = []

    def record_fit(*args, **kwargs):
        """Detecta ajustes indebidos sin detener el probe antes de comprobar el rechazo."""
        fit_calls.append(True)
        return original_fit(*args, **kwargs)

    original_fit = definitions["Pipeline"].fit
    monkeypatch.setattr(definitions["Pipeline"], "fit", record_fit)
    try:
        with pytest.raises(ValueError, match=re.escape(message)):
            definitions["run_task"](frame, "classification", destination)
    finally:
        assert not fit_calls, "Rejected split must not fit any pipeline"
        assert not destination.exists(), (
            "Rejected split must not create/export destination"
        )
        assert raw.equals(original)
        assert frame.height == 600


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_notebook_offline_end_to_end_exports(definitions, tmp_path, task):
    from test_pipeline import fixture_frame

    frame, _ = definitions["clean_bookings"](fixture_frame())
    result = definitions["run_task"](frame, task, tmp_path)
    assert len(result["metrics"]) == 2
    assert set(result["fingerprint_overlap"].values()) == {0}
    stored_ids = set()
    for split, details in result["splits"].items():
        stored = pl.read_parquet(tmp_path / f"{task}_{split}.parquet")
        assert stored.height == details["rows"]
        assert "adults" in stored.columns and "total_guests" in stored.columns
        assert not set(stored["row_id"]) & stored_ids
        stored_ids |= set(stored["row_id"])
        assert "reservation_status" not in stored.columns
        for metrics in result["metrics"].values():
            assert set(metrics) == {"validation", "test"}
    assert len(stored_ids) == result["population_rows"]
