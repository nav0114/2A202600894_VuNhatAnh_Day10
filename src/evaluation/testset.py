from __future__ import annotations

from typing import Any

import pandas as pd


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    from core.utils import write_json
    from pathlib import Path
    
    if len(df) < 1:
        raise ValueError("Cleaned DataFrame is empty. Cannot build test set.")
        
    test_set = []
    # Pick up to 5 representative papers
    sample_papers = df.head(min(5, len(df))).to_dict(orient="records")
    
    for idx, row in enumerate(sample_papers):
        paper_id = row["paper_id"]
        title = row["title"]
        
        # 1. Summary question
        test_set.append({
            "id": f"q_summary_{idx}",
            "question_type": "summary",
            "question": f"What is the summary of the paper '{title}'?",
            "ground_truth": row["summary"],
            "ground_truth_doc_ids": [paper_id]
        })
        
        # 2. Authors question
        test_set.append({
            "id": f"q_authors_{idx}",
            "question_type": "authors",
            "question": f"Who authored the paper '{title}'?",
            "ground_truth": row["authors_joined"],
            "ground_truth_doc_ids": [paper_id]
        })
        
        # 3. Date question
        test_set.append({
            "id": f"q_date_{idx}",
            "question_type": "date",
            "question": f"When was the paper '{title}' published?",
            "ground_truth": row["published"],
            "ground_truth_doc_ids": [paper_id]
        })
        
        # 4. Categories question
        if row["categories_joined"]:
            test_set.append({
                "id": f"q_categories_{idx}",
                "question_type": "categories",
                "question": f"What categories does the paper '{title}' belong to?",
                "ground_truth": row["categories_joined"],
                "ground_truth_doc_ids": [paper_id]
            })
            
    write_json(Path(output_path), test_set)
    return test_set
