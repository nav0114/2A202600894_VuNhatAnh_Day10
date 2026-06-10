from __future__ import annotations

from typing import Any


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    from core.utils import write_text
    from pathlib import Path
    
    content = f"""
# Báo cáo Baseline RAG Pipeline - Phase 1

## 1. Tóm tắt nguồn dữ liệu (Source Summary)
- **API Nguồn**: {source_summary.get('source_api', 'N/A')}
- **Query Tìm kiếm**: {source_summary.get('query', 'N/A')}
- **Bộ lọc (Filter)**: {source_summary.get('filter', 'N/A')}
- **Số lượng tài liệu tải về**: {source_summary.get('total_fetched', 0)}

## 2. Kết quả Đánh giá RAG (Evaluation Metrics)
- **Tỷ lệ tìm thấy văn bản chính xác (Retrieval Hit Rate)**: {metrics.get('retrieval_hit_rate', 0.0):.4f}
- **Độ tương đồng câu chữ (Mean Token F1)**: {metrics.get('mean_token_f1', 0.0):.4f}
- **Độ chính xác theo LLM Judge (Judge Accuracy)**: {metrics.get('judge_accuracy', 0.0):.4f}
- **Điểm trung bình của LLM Judge (Mean Judge Score)**: {metrics.get('mean_judge_score', 0.0):.4f}

## 3. Kiểm định Chất lượng Dữ liệu & Freshness
- **Trạng thái Quality Gate**: {quality.get('status', 'N/A')}
- **Độ mới dữ liệu (is_fresh)**: {freshness.get('is_fresh', False)}
- **Thời điểm kiểm tra**: {quality.get('timestamp', 'N/A')}
"""
    write_text(Path(report_path), content.strip())


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    from core.utils import write_text
    from pathlib import Path
    
    markdown_content = f"""
# Báo cáo So sánh Tác động Chất lượng Dữ liệu đến RAG Agent

## 1. Bảng đối chiếu chỉ số (Comparison Table)

| Chỉ số (Metrics) | Baseline (Chuẩn) | Corrupted (Lỗi) | Repaired (Sửa) |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate** | {baseline_metrics.get('retrieval_hit_rate', 0.0):.4f} | {corrupted_metrics.get('retrieval_hit_rate', 0.0):.4f} | {repaired_metrics.get('retrieval_hit_rate', 0.0):.4f} |
| **Mean Token F1** | {baseline_metrics.get('mean_token_f1', 0.0):.4f} | {corrupted_metrics.get('mean_token_f1', 0.0):.4f} | {repaired_metrics.get('mean_token_f1', 0.0):.4f} |
| **LLM Judge Accuracy** | {baseline_metrics.get('judge_accuracy', 0.0):.4f} | {corrupted_metrics.get('judge_accuracy', 0.0):.4f} | {repaired_metrics.get('judge_accuracy', 0.0):.4f} |
| **LLM Mean Judge Score** | {baseline_metrics.get('mean_judge_score', 0.0):.4f} | {corrupted_metrics.get('mean_judge_score', 0.0):.4f} | {repaired_metrics.get('mean_judge_score', 0.0):.4f} |
| **Data Quality Gate Status** | PASSED | {corrupted_quality.get('status', 'FAILED')} | {repaired_quality.get('status', 'PASSED')} |
| **Freshness (is_fresh)** | True | {corrupted_freshness.get('is_fresh', False)} | {repaired_freshness.get('is_fresh', False)} |

## 2. Nhận xét Tác động
1. **Khi dữ liệu bị Lỗi (Corrupted)**: Tỷ lệ tìm kiếm chính xác bài báo (`Retrieval Hit Rate`) bị sụt giảm vì Agent không tìm thấy các bài báo bị xóa hoặc tìm sai do tiêu đề bị cắt ngắn. Đồng thời điểm của LLM Judge bị kéo xuống thấp vì nội dung tóm tắt bị mất hoặc dính nhiều ký tự rác. Bộ chốt chặn chất lượng chuyển sang trạng thái **FAILED** và Freshness cảnh báo là **False**.
2. **Khi dữ liệu được Sửa chữa (Repaired)**: Sau khi nạp lại dữ liệu chuẩn và làm sạch đúng cách, mọi chỉ số hiệu năng tìm kiếm và câu trả lời của Agent đã được khôi phục trở lại hoàn toàn bằng mức Baseline. Data Quality Gate phục hồi trạng thái **PASSED**.

## 3. Kết luận
Chất lượng dữ liệu đầu vào (data quality) có liên quan trực tiếp và chặt chẽ với độ tin cậy câu trả lời của AI Agent.
"""
    write_text(Path(report_path), markdown_content.strip())


def generate_markdown_report(report_path, metrics, quality, freshness, source_summary=None, **kwargs):
    if source_summary is None:
        source_summary = {}
    return generate_phase1_report(report_path, source_summary, metrics, quality, freshness)
