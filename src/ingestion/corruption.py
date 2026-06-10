from __future__ import annotations

import pandas as pd


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path=None, output_path=None, manifest_path=None, **kwargs) -> pd.DataFrame:
    import json
    from core.utils import ensure_parent
    from pathlib import Path
    
    log_path = output_log_path or output_path or manifest_path
    corrupted_df = df.copy()
    
    # 1. Drop 5 records mới nhất
    dropped_ids = list(corrupted_df.iloc[:5]["paper_id"]) if len(corrupted_df) >= 5 else []
    corrupted_df = corrupted_df.iloc[len(dropped_ids):].copy().reset_index(drop=True)
    
    # 2. Xóa summary của 3 dòng đầu
    blanked_ids = list(corrupted_df.iloc[:3]["paper_id"]) if len(corrupted_df) >= 3 else []
    for i in range(min(3, len(corrupted_df))):
        corrupted_df.iloc[i, corrupted_df.columns.get_loc("summary")] = ""
        
    # 3. Inject noise vào summary của 3 dòng tiếp theo
    noised_ids = list(corrupted_df.iloc[3:6]["paper_id"]) if len(corrupted_df) >= 6 else []
    for i in range(min(3, len(corrupted_df) - 3)):
        idx = i + 3
        corrupted_df.iloc[idx, corrupted_df.columns.get_loc("summary")] += " [CORRUPTED NOISE_XYZ]"
        
    # 4. Truncate title còn 3 từ
    truncated_ids = list(corrupted_df.iloc[:3]["paper_id"]) if len(corrupted_df) >= 3 else []
    for i in range(min(3, len(corrupted_df))):
        title_val = corrupted_df.iloc[i]["title"]
        corrupted_df.iloc[i, corrupted_df.columns.get_loc("title")] = " ".join(title_val.split()[:3])
        
    # 5. Làm stale published date của 2 dòng đầu
    stale_ids = list(corrupted_df.iloc[:2]["paper_id"]) if len(corrupted_df) >= 2 else []
    for i in range(min(2, len(corrupted_df))):
        corrupted_df.iloc[i, corrupted_df.columns.get_loc("published")] = "1990-01-01"
        corrupted_df.iloc[i, corrupted_df.columns.get_loc("age_days")] = 10000
        
    # 6. Thêm bản ghi trùng lặp
    if not corrupted_df.empty:
        first_row = corrupted_df.iloc[[0]]
        corrupted_df = pd.concat([corrupted_df, first_row, first_row], ignore_index=True)
        
    # 7. Rebuild text_for_embedding
    corrupted_df["text_for_embedding"] = corrupted_df.apply(
        lambda r: f"Title: {r['title']}\nSummary: {r['summary']}", axis=1
    )
    
    # 8. Ghi log
    log_data = {
        "dropped_records": dropped_ids,
        "blanked_summaries": blanked_ids,
        "noised_summaries": noised_ids,
        "truncated_titles": truncated_ids,
        "stale_dates": stale_ids,
        "duplicates_added": 2
    }
    
    if log_path:
        ensure_parent(Path(log_path))
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
    return corrupted_df
