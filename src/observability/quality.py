from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import Settings


def run_data_quality_checks(
    df: pd.DataFrame, 
    settings: Settings | None = None, 
    report_name: str = "quality", 
    output_path = None,
    report_path = None,
    **kwargs
) -> dict[str, Any]:
    from datetime import UTC, datetime
    from core.utils import write_json
    from pathlib import Path
    
    total_rows = len(df)
    null_paper_ids = int(df["paper_id"].isna().sum() + (df["paper_id"].astype(str).str.strip() == "").sum())
    duplicate_paper_ids = int(df.duplicated(subset=["paper_id"]).sum())
    null_titles = int(df["title"].isna().sum() + (df["title"].astype(str).str.strip() == "").sum())
    
    # Check length of summaries (short abstract is a quality issue)
    short_summaries = int((df["summary"].astype(str).str.len() < 100).sum())
    
    # Check freshness (age_days > threshold)
    threshold = settings.freshness_threshold_days if settings else 180
    stale_rows = int((df["age_days"] > threshold).sum())
    
    status = "PASSED"
    failures = []
    
    if total_rows == 0:
        status = "FAILED"
        failures.append("Dataset is empty.")
    if null_paper_ids > 0:
        status = "FAILED"
        failures.append(f"Found {null_paper_ids} null or empty paper_ids.")
    if duplicate_paper_ids > 0:
        status = "FAILED"
        failures.append(f"Found {duplicate_paper_ids} duplicate paper_ids.")
    if null_titles > 0:
        status = "FAILED"
        failures.append(f"Found {null_titles} null or empty titles.")
        
    report = {
        "report_name": report_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
        "failures": failures,
        "metrics": {
            "total_rows": total_rows,
            "null_paper_ids": null_paper_ids,
            "duplicate_paper_ids": duplicate_paper_ids,
            "null_titles": null_titles,
            "short_summaries": short_summaries,
            "stale_rows": stale_rows,
        }
    }
    
    log_file = output_path or report_path
    if log_file:
        write_json(Path(log_file), report)
    elif settings:
        report_file = settings.paths.quality_dir / f"{report_name}_quality_checks.json"
        write_json(report_file, report)
        
    return report


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    from core.utils import write_json
    
    total_rows = len(df)
    valid_published = df[df["published"].astype(str).str.strip() != ""]["published"]
    
    if not valid_published.empty:
        latest_published = str(valid_published.max())
        oldest_published = str(valid_published.min())
    else:
        latest_published = "N/A"
        oldest_published = "N/A"
        
    threshold = settings.freshness_threshold_days
    stale_rows = int((df["age_days"] > threshold).sum())
    
    is_fresh = bool(not df.empty and int(df["age_days"].min()) <= threshold)
    
    report = {
        "latest_published": latest_published,
        "oldest_published": oldest_published,
        "stale_rows": stale_rows,
        "total_rows": total_rows,
        "is_fresh": is_fresh,
    }
    
    write_json(report_path, report)
    return report


run_quality_checks = run_data_quality_checks
