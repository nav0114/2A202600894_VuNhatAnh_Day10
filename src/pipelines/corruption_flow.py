from datetime import UTC, datetime

from core.config import load_settings
from core.utils import read_json, write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.corruption import corrupt_clean_dataframe
from ingestion.crossref import load_raw_records
from retrieval.index import LocalEmbeddingIndex
from observability.quality import run_data_quality_checks, build_freshness_report
from observability.reporting import generate_corruption_report


# compare pipelines flow
def main() -> None:
    settings = load_settings()
    
    # 1. Load baseline metrics va clean dataset
    print("Loading baseline metrics and raw dataset...")
    baseline_metrics = read_json(settings.paths.baseline_metrics)
    raw_records = load_raw_records(settings.paths.raw_records_json)
    df_clean = build_clean_dataframe(raw_records, datetime.now(UTC))
    
    # 2. Tao corrupted dataframe
    print("Simulating data corruption (Corrupting)...")
    df_corrupted = corrupt_clean_dataframe(df_clean, settings.paths.corruption_log)
    
    # 3. Save corrupted artifacts
    write_csv(df_corrupted, settings.paths.corrupted_clean_csv)
    write_json(settings.paths.corrupted_clean_json, df_corrupted.to_dict(orient="records"))
    
    # 4. Rebuild index va evaluate
    print("Building index for corrupted dataset...")
    index_corrupted = LocalEmbeddingIndex.build(df_corrupted, settings, settings.paths.corrupted_embeddings_json)
    
    print("Evaluating agent performance on corrupted dataset...")
    eval_corrupted = evaluate_pipeline(
        settings=settings,
        index=index_corrupted,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.corrupted_metrics,
        answers_output_path=settings.paths.corrupted_answers
    )
    
    # 5. Run quality checks/freshness tren corrupted data
    quality_corrupted = run_data_quality_checks(df_corrupted, settings, "corrupted")
    freshness_corrupted = build_freshness_report(
        df_corrupted, 
        settings, 
        settings.paths.quality_dir / "corrupted_freshness_report.json"
    )
    
    # 6. Repair lai tu raw records (tai thiet dataframe sach)
    print("Re-ingesting and cleaning raw records (Repairing)...")
    df_repaired = build_clean_dataframe(raw_records, datetime.now(UTC))
    write_csv(df_repaired, settings.paths.repaired_clean_csv)
    write_json(settings.paths.repaired_clean_json, df_repaired.to_dict(orient="records"))
    
    # 7. Evaluate repaired dataset
    print("Building index for repaired dataset...")
    index_repaired = LocalEmbeddingIndex.build(df_repaired, settings, settings.paths.repaired_embeddings_json)
    
    print("Evaluating agent performance on repaired dataset...")
    eval_repaired = evaluate_pipeline(
        settings=settings,
        index=index_repaired,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.repaired_metrics,
        answers_output_path=settings.paths.repaired_answers
    )
    
    quality_repaired = run_data_quality_checks(df_repaired, settings, "repaired")
    freshness_repaired = build_freshness_report(
        df_repaired, 
        settings, 
        settings.paths.quality_dir / "repaired_freshness_report.json"
    )
    
    # 8. Tao comparison report
    print("Generating corruption comparison report...")
    generate_corruption_report(
        report_path=settings.paths.comparison_report,
        baseline_metrics=baseline_metrics,
        corrupted_metrics=eval_corrupted.summary,
        repaired_metrics=eval_repaired.summary,
        corrupted_quality=quality_corrupted,
        repaired_quality=quality_repaired,
        corrupted_freshness=freshness_corrupted,
        repaired_freshness=freshness_repaired
    )
    print("🎉 Corruption Comparison Flow Completed Successfully!")
