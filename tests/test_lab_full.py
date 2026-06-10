"""
Comprehensive pytest suite for the RAG data-pipeline lab.

Place this file at: tests/test_lab.py
Run from project root:
    uv run pytest -q
or:
    pytest -q

The tests follow Guide.md and intentionally avoid real network/API calls.
They validate contracts for:
- Crossref parsing
- Cleaning and freshness fields
- Evaluation test-set schema
- Corruption behavior and manifest/report output
- Quality/freshness/reporting contracts, when implemented
- Pipeline module presence and expected public entrypoints
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def import_required(module_name: str):
    """Import a module that is required by the lab."""
    return importlib.import_module(module_name)


def import_optional(module_name: str):
    """Import an optional/advanced module; skip the test if it is absent."""
    return pytest.importorskip(module_name)


def first_existing_attr(module: Any, names: list[str]) -> Callable[..., Any]:
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    pytest.fail(
        f"{module.__name__} must expose one of these callables: {', '.join(names)}"
    )


def record_to_dict(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if hasattr(record, "dict"):
        return record.dict()
    if isinstance(record, dict):
        return record
    return {
        key: getattr(record, key)
        for key in dir(record)
        if not key.startswith("_") and not callable(getattr(record, key))
    }


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call func with only kwargs its signature accepts."""
    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**supported)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_date() -> datetime:
    return datetime(2025, 12, 15, tzinfo=UTC)


@pytest.fixture
def rich_crossref_payload() -> dict[str, Any]:
    """A small Crossref-like payload with valid, edge, and invalid records."""
    return {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/test.doi.1",
                    "title": ["  Test Paper Title One  "],
                    "abstract": "<jats:p>This is a test abstract containing XML tags.</jats:p>",
                    "author": [
                        {"given": "First", "family": "Last"},
                        {"given": "Second", "family": "Author"},
                    ],
                    "subject": ["Computer Science", "Information Retrieval"],
                    "published-online": {"date-parts": [[2025, 12, 10]]},
                    "URL": "https://doi.org/10.1234/test.doi.1",
                    "type": "journal-article",
                },
                {
                    "DOI": "10.1234/test.doi.2",
                    "title": ["Fresh RAG Evaluation Paper"],
                    "abstract": "<jats:p>RAG evaluation uses retrieval and answer metrics.</jats:p>",
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "subject": ["Artificial Intelligence"],
                    "published-print": {"date-parts": [[2025, 12, 14]]},
                    "URL": "https://doi.org/10.1234/test.doi.2",
                    "type": "proceedings-article",
                },
                {
                    # Invalid/low-quality record: cleaning should remove it if parsing keeps it.
                    "DOI": "10.1234/invalid.no.summary",
                    "title": [""],
                    "abstract": "",
                    "author": [],
                    "subject": [],
                    "published-online": {"date-parts": [[2020, 1, 1]]},
                    "URL": "https://doi.org/10.1234/invalid.no.summary",
                    "type": "journal-article",
                },
            ]
        }
    }


@pytest.fixture
def parsed_records(rich_crossref_payload: dict[str, Any]):
    crossref = import_required("ingestion.crossref")
    parse_crossref_payload = first_existing_attr(crossref, ["parse_crossref_payload"])
    return parse_crossref_payload(rich_crossref_payload)


@pytest.fixture
def clean_df(parsed_records: list[Any], run_date: datetime) -> pd.DataFrame:
    cleaning = import_required("ingestion.cleaning")
    build_clean_dataframe = first_existing_attr(cleaning, ["build_clean_dataframe"])
    return call_with_supported_kwargs(
        build_clean_dataframe,
        records=parsed_records,
        raw_records=parsed_records,
        run_date=run_date,
        as_of_date=run_date,
    )


# ---------------------------------------------------------------------------
# Step 2: Required modules should exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "core.config",
        "ingestion.crossref",
        "ingestion.cleaning",
        "retrieval.embeddings",
        "retrieval.index",
        "retrieval.agent",
        "retrieval.qa",
        "retrieval.llm",
        "evaluation.testset",
        "evaluation.metrics",
        "observability.quality",
        "observability.reporting",
        "pipelines.phase1",
        "pipelines.corruption_flow",
    ],
)
def test_required_lab_modules_are_importable(module_name: str):
    import_required(module_name)


# ---------------------------------------------------------------------------
# Step 3: Crossref parsing contract
# ---------------------------------------------------------------------------


def test_parse_crossref_payload_returns_consistent_schema(parsed_records: list[Any]):
    assert isinstance(parsed_records, list)
    assert len(parsed_records) >= 2

    first = record_to_dict(parsed_records[0])
    required_fields = {
        "paper_id",
        "title",
        "summary",
        "authors",
        "categories",
        "primary_category",
        "published",
    }
    assert required_fields.issubset(first.keys())

    assert first["paper_id"] == "10.1234/test.doi.1"
    assert first["title"] == "Test Paper Title One"
    assert first["summary"] == "This is a test abstract containing XML tags."
    assert first["authors"] == ["First Last", "Second Author"]
    assert first["categories"][:2] == ["Computer Science", "Information Retrieval"]
    assert first["primary_category"] == "Computer Science"
    assert first["published"] == "2025-12-10"


def test_parse_crossref_payload_supports_print_or_online_dates(parsed_records: list[Any]):
    second = record_to_dict(parsed_records[1])
    assert second["paper_id"] == "10.1234/test.doi.2"
    assert second["published"] == "2025-12-14"


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},
        {"message": {}},
        {"message": {"items": None}},
        {"message": {"items": []}},
    ],
)
def test_parse_crossref_payload_handles_empty_or_bad_payloads(bad_payload: dict[str, Any]):
    crossref = import_required("ingestion.crossref")
    parse_crossref_payload = first_existing_attr(crossref, ["parse_crossref_payload"])
    records = parse_crossref_payload(bad_payload)
    assert records == []


# ---------------------------------------------------------------------------
# Step 4: Cleaning contract
# ---------------------------------------------------------------------------


def test_build_clean_dataframe_core_columns_and_freshness(clean_df: pd.DataFrame):
    assert isinstance(clean_df, pd.DataFrame)
    assert len(clean_df) >= 2

    required_columns = {
        "paper_id",
        "title",
        "summary",
        "authors",
        "categories",
        "primary_category",
        "published",
        "age_days",
        "summary_chars",
        "text_for_embedding",
    }
    assert required_columns.issubset(set(clean_df.columns))

    row = clean_df.loc[clean_df["paper_id"] == "10.1234/test.doi.1"].iloc[0]
    assert row["title"] == "Test Paper Title One"
    assert row["summary"] == "This is a test abstract containing XML tags."
    assert row["summary_chars"] == len(row["summary"])
    assert int(row["age_days"]) == 5
    assert "Title: Test Paper Title One" in row["text_for_embedding"]
    assert "Summary: This is a test abstract" in row["text_for_embedding"]


def test_build_clean_dataframe_removes_invalid_or_empty_records(clean_df: pd.DataFrame):
    assert "10.1234/invalid.no.summary" not in set(clean_df["paper_id"])
    assert clean_df["paper_id"].notna().all()
    assert clean_df["title"].astype(str).str.strip().ne("").all()
    assert clean_df["summary"].astype(str).str.strip().ne("").all()


def test_build_clean_dataframe_deduplicates_by_paper_id(parsed_records: list[Any], run_date: datetime):
    cleaning = import_required("ingestion.cleaning")
    build_clean_dataframe = first_existing_attr(cleaning, ["build_clean_dataframe"])
    duplicated_records = parsed_records + [parsed_records[0]]

    df = call_with_supported_kwargs(
        build_clean_dataframe,
        records=duplicated_records,
        raw_records=duplicated_records,
        run_date=run_date,
        as_of_date=run_date,
    )

    assert df["paper_id"].is_unique


# ---------------------------------------------------------------------------
# Step 5: Evaluation test set contract
# ---------------------------------------------------------------------------


def test_build_test_set_schema_and_persistence(clean_df: pd.DataFrame, tmp_path: Path):
    testset_module = import_required("evaluation.testset")
    build_test_set = first_existing_attr(testset_module, ["build_test_set"])
    out_path = tmp_path / "test_set.json"

    test_set = call_with_supported_kwargs(
        build_test_set,
        df=clean_df,
        clean_df=clean_df,
        output_path=out_path,
        path=out_path,
    )

    assert isinstance(test_set, list)
    assert len(test_set) > 0

    sample = test_set[0]
    required_keys = {"question", "ground_truth", "ground_truth_doc_ids", "question_type"}
    assert required_keys.issubset(sample.keys())
    assert isinstance(sample["question"], str) and sample["question"].strip()
    assert isinstance(sample["ground_truth"], str) and sample["ground_truth"].strip()
    assert isinstance(sample["ground_truth_doc_ids"], list)
    assert sample["ground_truth_doc_ids"]
    assert sample["question_type"] in {"summary", "factual", "category", "freshness", "lookup"}

    assert out_path.exists(), "build_test_set should persist the generated test set"
    persisted = read_json(out_path)
    assert isinstance(persisted, list)
    assert persisted[0]["ground_truth_doc_ids"] == sample["ground_truth_doc_ids"]


# ---------------------------------------------------------------------------
# Step 10: Metrics contract
# ---------------------------------------------------------------------------


def test_metrics_module_exposes_required_metric_contracts():
    metrics = import_required("evaluation.metrics")
    available = {name for name in dir(metrics) if not name.startswith("_")}

    assert available.intersection(
        {
            "evaluate",
            "evaluate_agent",
            "evaluate_predictions",
            "compute_metrics",
            "calculate_metrics",
        }
    ), "evaluation.metrics should expose an evaluation entrypoint"

    source = inspect.getsource(metrics)
    for metric_name in [
        "retrieval_hit_rate",
        "mean_token_f1",
        "judge_accuracy",
        "mean_judge_score",
    ]:
        assert metric_name in source, f"metrics output should include {metric_name}"


# ---------------------------------------------------------------------------
# Step 11: Data quality and reporting contracts
# ---------------------------------------------------------------------------


def test_quality_checks_return_machine_readable_results(clean_df: pd.DataFrame, tmp_path: Path):
    quality = import_required("observability.quality")
    quality_fn = first_existing_attr(
        quality,
        [
            "run_quality_checks",
            "check_data_quality",
            "build_quality_report",
            "generate_quality_report",
        ],
    )

    result = call_with_supported_kwargs(
        quality_fn,
        df=clean_df,
        clean_df=clean_df,
        output_path=tmp_path / "quality.json",
        report_path=tmp_path / "quality.json",
    )

    assert result is not None
    if isinstance(result, pd.DataFrame):
        assert not result.empty
    elif isinstance(result, dict):
        flattened = json.dumps(result).lower()
        assert any(word in flattened for word in ["pass", "fail", "success", "fresh", "quality"])
    elif isinstance(result, list):
        assert len(result) > 0
    else:
        assert hasattr(result, "__dict__")


def test_reporting_module_can_create_markdown_report(tmp_path: Path):
    reporting = import_required("observability.reporting")
    report_fn = first_existing_attr(
        reporting,
        [
            "write_markdown_report",
            "generate_markdown_report",
            "create_report",
            "build_report",
        ],
    )

    out_path = tmp_path / "report.md"
    result = call_with_supported_kwargs(
        report_fn,
        output_path=out_path,
        report_path=out_path,
        title="Lab Report",
        metrics={
            "retrieval_hit_rate": 1.0,
            "mean_token_f1": 0.8,
            "judge_accuracy": 1.0,
            "mean_judge_score": 4.0,
        },
        quality={"status": "pass"},
        freshness={"status": "fresh"},
    )

    if out_path.exists():
        content = out_path.read_text(encoding="utf-8")
    else:
        content = str(result)

    assert "retrieval" in content.lower() or "quality" in content.lower()
    assert "#" in content or "report" in content.lower()


# ---------------------------------------------------------------------------
# Step 12: Corruption contract
# ---------------------------------------------------------------------------


def test_corrupt_clean_dataframe_changes_data_and_writes_manifest(clean_df: pd.DataFrame, tmp_path: Path):
    corruption = import_required("ingestion.corruption")
    corrupt_clean_dataframe = first_existing_attr(corruption, ["corrupt_clean_dataframe"])
    manifest_path = tmp_path / "corruption_manifest.json"

    corrupted_df = call_with_supported_kwargs(
        corrupt_clean_dataframe,
        df=clean_df,
        clean_df=clean_df,
        output_path=manifest_path,
        manifest_path=manifest_path,
        random_seed=42,
        seed=42,
    )

    assert isinstance(corrupted_df, pd.DataFrame)
    assert len(corrupted_df) > 0
    assert set(clean_df.columns).issubset(set(corrupted_df.columns))

    # The corruption step should have a measurable impact: changed row count,
    # changed text/title/summary/date, duplicates, or missing summary values.
    same_shape = corrupted_df.shape == clean_df.shape
    same_core_content = False
    if same_shape:
        core_cols = [c for c in ["paper_id", "title", "summary", "published"] if c in clean_df.columns]
        same_core_content = corrupted_df[core_cols].reset_index(drop=True).equals(
            clean_df[core_cols].reset_index(drop=True)
        )

    has_duplicates = "paper_id" in corrupted_df.columns and corrupted_df["paper_id"].duplicated().any()
    has_blank_summary = "summary" in corrupted_df.columns and corrupted_df["summary"].fillna("").astype(str).str.strip().eq("").any()

    assert not (same_shape and same_core_content) or has_duplicates or has_blank_summary

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest, "corruption manifest should not be empty"


# ---------------------------------------------------------------------------
# Steps 9 and 13: Pipeline entrypoint contracts without executing APIs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, expected_names",
    [
        (
            "pipelines.phase1",
            ["run_phase1", "run", "main", "baseline_pipeline", "run_baseline_pipeline"],
        ),
        (
            "pipelines.corruption_flow",
            ["run_corruption_flow", "run", "main", "corruption_flow"],
        ),
    ],
)
def test_pipeline_modules_expose_runnable_entrypoints(module_name: str, expected_names: list[str]):
    module = import_required(module_name)
    first_existing_attr(module, expected_names)


def test_phase1_source_mentions_required_outputs():
    phase1 = import_required("pipelines.phase1")
    source = inspect.getsource(phase1)
    for required_fragment in [
        "clean",
        "embedding",
        "eval",
        "result",
        "quality",
        "report",
    ]:
        assert required_fragment in source.lower()


def test_corruption_flow_source_mentions_repair_and_comparison():
    flow = import_required("pipelines.corruption_flow")
    source = inspect.getsource(flow).lower()
    for required_fragment in ["corrupt", "repair", "compare"]:
        assert required_fragment in source


# ---------------------------------------------------------------------------
# Step 6 and 8: Retrieval and agent contracts without downloading models
# ---------------------------------------------------------------------------


def test_retrieval_index_exposes_build_and_query_contracts():
    index = import_required("retrieval.index")
    available = {name for name in dir(index) if not name.startswith("_")}

    assert available.intersection(
        {"build_index", "build_vector_store", "create_index", "index_documents"}
    ), "retrieval.index should expose an index-building function"

    assert available.intersection(
        {"query_index", "search", "semantic_search", "retrieve", "query"}
    ), "retrieval.index should expose a top-k query/retrieval function"


def test_agent_or_qa_exposes_answer_contract():
    agent = import_required("retrieval.agent")
    qa = import_required("retrieval.qa")
    available = {name for name in dir(agent) if not name.startswith("_")} | {
        name for name in dir(qa) if not name.startswith("_")
    }

    assert available.intersection(
        {"answer_question", "ask", "query_agent", "run_agent", "RAGAgent", "QAAgent"}
    ), "retrieval.agent/retrieval.qa should expose a question-answering contract"


# ---------------------------------------------------------------------------
# Step 7: LLM provider configuration contract
# ---------------------------------------------------------------------------


def test_llm_module_mentions_required_providers_and_env_keys():
    llm = import_required("retrieval.llm")
    source = inspect.getsource(llm).lower()

    for provider in ["openai", "gemini", "anthropic", "openrouter", "ollama"]:
        assert provider in source

    assert "llm_provider" in source
    assert "llm_model" in source


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------


def test_load_settings_exposes_data_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = import_required("core.config")
    load_settings = first_existing_attr(config, ["load_settings"])

    # Some implementations read env vars; point them at a temp root if supported.
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    settings = load_settings()
    settings_dict = record_to_dict(settings)
    flattened = json.dumps(settings_dict, default=str).lower()

    for expected in ["raw", "clean", "eval", "result", "quality", "report"]:
        assert expected in flattened
