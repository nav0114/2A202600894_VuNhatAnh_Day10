from __future__ import annotations

from datetime import datetime

import pandas as pd

from ingestion.crossref import PaperRecord


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    import re
    from dataclasses import asdict
    
    if not records:
        return pd.DataFrame(columns=[
            "paper_id", "title", "summary", "authors", "categories", "primary_category",
            "published", "updated", "abs_url", "pdf_url", "comment",
            "authors_joined", "categories_joined", "summary_chars", "text_for_embedding", "age_days"
        ])
        
    # Convert list of PaperRecord to DataFrame
    df = pd.DataFrame([asdict(rec) for rec in records])
    
    # 1. Normalize title, summary, authors, categories
    df["title"] = df["title"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notna(x) else "")
    df["summary"] = df["summary"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notna(x) else "")
    
    df["authors"] = df["authors"].apply(lambda authors: [str(a).strip() for a in authors if str(a).strip()] if isinstance(authors, list) else [])
    df["categories"] = df["categories"].apply(lambda cats: [str(c).strip() for c in cats if str(c).strip()] if isinstance(cats, list) else [])
    
    # 2. Parse published/updated date & 3. Calculate age_days
    run_date_naive = run_date.replace(tzinfo=None)
    
    def calculate_age(date_str):
        if not date_str:
            return 9999
        try:
            pub_date = pd.to_datetime(date_str).replace(tzinfo=None)
            delta = run_date_naive - pub_date
            return max(0, int(delta.days))
        except Exception:
            return 9999
            
    df["age_days"] = df["published"].apply(calculate_age)
    
    # 4. Create helper columns
    df["authors_joined"] = df["authors"].apply(lambda authors: ", ".join(authors))
    df["categories_joined"] = df["categories"].apply(lambda cats: ", ".join(cats))
    df["summary_chars"] = df["summary"].apply(len)
    
    # Create text_for_embedding by joining Title and Summary
    df["text_for_embedding"] = df.apply(lambda row: f"Title: {row['title']}\nSummary: {row['summary']}", axis=1)
    
    # 5. Drop duplicates and filter bad rows
    df = df.drop_duplicates(subset=["paper_id"], keep="first")
    df = df[df["paper_id"].str.strip() != ""]
    df = df[df["title"].str.strip() != ""]
    df = df[df["summary"].str.strip() != ""]
    
    # 6. Sort dataframe and return
    df = df.sort_values(by=["published", "paper_id"], ascending=[False, True]).reset_index(drop=True)
    
    return df
