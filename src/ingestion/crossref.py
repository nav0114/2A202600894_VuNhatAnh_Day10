from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config import Settings


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    import html
    import re
    
    if not isinstance(payload, dict):
        return []
    message = payload.get("message") or {}
    items = message.get("items") or []
    records = []
    
    for item in items:
        # 1. DOI
        paper_id = item.get("DOI", "").strip()
        if not paper_id:
            continue
            
        # 2. Title
        title_list = item.get("title", [])
        if not title_list or not title_list[0].strip():
            continue
        title = re.sub(r"\s+", " ", title_list[0]).strip()
        
        # 3. Abstract / Summary
        abstract = item.get("abstract", "").strip()
        if not abstract:
            continue
        # Clean JATS XML tags
        summary = re.sub(r"<[^>]+>", "", abstract)
        summary = html.unescape(summary)
        summary = re.sub(r"\s+", " ", summary).strip()
        if not summary:
            continue
            
        # 4. Authors
        authors = []
        for author_info in item.get("author", []):
            given = author_info.get("given", "").strip()
            family = author_info.get("family", "").strip()
            if given and family:
                authors.append(f"{given} {family}")
            elif given:
                authors.append(given)
            elif family:
                authors.append(family)
                
        # 5. Categories
        categories = item.get("subject", [])
        categories = [re.sub(r"\s+", " ", cat).strip() for cat in categories if cat.strip()]
        primary_category = categories[0] if categories else "N/A"
        
        # 6. Published date
        published = ""
        # Try published-online, then published-print, then issued, then created
        for field in ["published-online", "published-print", "issued", "created"]:
            date_info = item.get(field)
            if date_info and "date-parts" in date_info:
                parts = date_info["date-parts"]
                if parts and len(parts[0]) > 0:
                    date_list = parts[0]
                    if len(date_list) == 3:
                        published = f"{date_list[0]:04d}-{date_list[1]:02d}-{date_list[2]:02d}"
                        break
                    elif len(date_list) == 2:
                        published = f"{date_list[0]:04d}-{date_list[1]:02d}-01"
                        break
                    elif len(date_list) == 1:
                        published = f"{date_list[0]:04d}-01-01"
                        break
        if not published:
            created_info = item.get("created")
            if created_info and "date-time" in created_info:
                published = created_info["date-time"][:10]
            else:
                published = "2026-06-10"
                
        updated = published
        
        # 7. URLs
        abs_url = item.get("URL", "").strip()
        if not abs_url:
            abs_url = f"https://doi.org/{paper_id}"
            
        pdf_url = ""
        for link in item.get("link", []):
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL", "").strip()
                break
        if not pdf_url:
            pdf_url = abs_url
            
        # 8. Comment
        comment = item.get("type", "")
        
        record = PaperRecord(
            paper_id=paper_id,
            title=title,
            summary=summary,
            authors=authors,
            categories=categories,
            primary_category=primary_category,
            published=published,
            updated=updated,
            abs_url=abs_url,
            pdf_url=pdf_url,
            comment=comment,
        )
        records.append(record)
        
    return records


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    import requests
    import time
    from dataclasses import asdict
    from core.utils import write_json
    
    url = "https://api.crossref.org/works"
    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
    }
    
    headers = {
        "User-Agent": "Day10Lab/1.0 (mailto:student@example.com)"
    }
    
    max_retries = 3
    backoff_factor = 2
    response = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                break
            elif response.status_code in (429, 503, 504):
                time.sleep(backoff_factor ** attempt)
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff_factor ** attempt)
            
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "No Response"
        raise RuntimeError(f"Failed to fetch source records from Crossref API (Status code: {status})")
        
    payload = response.json()
    
    # Save raw API response
    write_json(settings.paths.raw_api_response, payload)
    
    # Parse records
    records = parse_crossref_payload(payload)
    
    # Save parsed records
    records_dict = [asdict(rec) for rec in records]
    write_json(settings.paths.raw_records_json, records_dict)
    
    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    from core.utils import read_json
    
    if not path.exists():
        return []
        
    try:
        data = read_json(path)
    except Exception:
        return []
        
    records = []
    for item in data:
        records.append(
            PaperRecord(
                paper_id=item.get("paper_id", ""),
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                authors=item.get("authors", []),
                categories=item.get("categories", []),
                primary_category=item.get("primary_category", "N/A"),
                published=item.get("published", ""),
                updated=item.get("updated", ""),
                abs_url=item.get("abs_url", ""),
                pdf_url=item.get("pdf_url", ""),
                comment=item.get("comment", ""),
            )
        )
    return records
