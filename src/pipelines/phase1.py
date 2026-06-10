from datetime import UTC, datetime

from core.config import load_settings
from core.utils import write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import fetch_source_records, load_raw_records
from retrieval.index import LocalEmbeddingIndex
from observability.quality import run_data_quality_checks, build_freshness_report
from observability.reporting import generate_phase1_report


def main() -> None:
    # 1. Load settings
    settings = load_settings()
    
    # 2. Load hoac fetch raw records
    raw_path = settings.paths.raw_records_json
    if settings.refresh_source or not raw_path.exists():
        print("Fetching raw records from source API...")
        records = fetch_source_records(settings)
    else:
        print("Loading raw records from disk...")
        records = load_raw_records(raw_path)
    print(f"Loaded {len(records)} raw records.")
    
    # 3. Clean data
    print("Cleaning records and building DataFrame...")
    run_date = datetime.now(UTC)
    df_clean = build_clean_dataframe(records, run_date)
    print(f"Cleaned DataFrame shape: {df_clean.shape}")
    
    # 4. Save clean CSV/JSON
    print("Saving cleaned dataset...")
    write_csv(df_clean, settings.paths.clean_csv)
    write_json(settings.paths.clean_json, df_clean.to_dict(orient="records"))
    
    # 5. Build Chroma index
    print("Building Chroma vector index...")
    index = LocalEmbeddingIndex.build(df_clean, settings)
    
    # 6. Tao hoac load evaluation set
    test_set_path = settings.paths.eval_testset
    if settings.refresh_test_set or not test_set_path.exists():
        print("Generating new evaluation test set...")
        test_set = build_test_set(df_clean, test_set_path)
    else:
        print("Using existing evaluation test set.")
        
    # 7. Evaluate
    print("Evaluating pipeline on the test set...")
    eval_bundle = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=test_set_path,
        metrics_output_path=settings.paths.baseline_metrics,
        answers_output_path=settings.paths.baseline_answers,
    )
    metrics_summary = eval_bundle.summary
    print(f"Evaluation complete. Hit rate: {metrics_summary.get('retrieval_hit_rate')}, Token F1: {metrics_summary.get('mean_token_f1')}")
    
    # 8. Run quality checks va freshness report
    print("Running data quality checks and freshness report...")
    quality_result = run_data_quality_checks(df_clean, settings, "baseline")
    freshness_result = build_freshness_report(df_clean, settings, settings.paths.freshness_report)
    
    # 9. Tao markdown report
    print("Generating Phase 1 markdown report...")
    source_summary = {
        "total_fetched": len(records),
        "source_api": settings.source_api,
        "query": settings.source_query,
        "filter": settings.source_filter,
    }
    generate_phase1_report(
        report_path=settings.paths.baseline_report,
        source_summary=source_summary,
        metrics=metrics_summary,
        quality=quality_result,
        freshness=freshness_result,
    )
    print("🎉 Phase 1 Pipeline Completed Successfully!")
