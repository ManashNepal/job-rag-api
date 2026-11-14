import logging
import re
import uuid

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "job_title",
    "company",
    "job_location",
    "job_link",
    "search_country",
    "job level",
    "job_type",
    "job_summary",
    "job_skills",
]

# Seniority from the job title. Checked in order, the first match wins
SENIORITY_LEVELS = ("intern", "junior", "mid", "senior")
_ROLE = r"\b(engineer|eng|developer|programmer)"
SENIORITY_RULES = [
    ("intern", re.compile(r"\b(intern|internship|co-?op)\b")),
    ("junior", re.compile(r"\b(junior|jr|entry[- ]level|graduate|grad)\b")),
    (
        "senior",
        re.compile(
            rf"\b(senior|sr|lead|staff|principal|manager|director|iii|iv)\b|{_ROLE}\s*(3|4)\b"
        ),
    ),
    ("mid", re.compile(rf"\b(mid|mid-level|ii)\b|{_ROLE}\s*2\b")),
]


def seniority_from_title(title: str) -> str:
    """Seniority level from title keywords, or "unspecified" if the title has none."""
    normalized = title.lower().replace(".", " ")
    for level, pattern in SENIORITY_RULES:
        if pattern.search(normalized):
            return level
    return "unspecified"


# Field cleaning
def clean_text(value) -> str:
    """Strip a value and collapse whitespace, keeping paragraph breaks."""
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_skills(raw) -> list[str]:
    """Split a comma-separated skills string into a de-duplicated list."""
    skills, seen = [], set()
    for skill in clean_text(raw).split(","):
        skill = skill.strip()
        if skill and skill.lower() not in seen:
            seen.add(skill.lower())
            skills.append(skill)
    return skills


def job_id_from_link(job_link: str) -> str:
    """Deterministic UUID for a job, so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, job_link))


# Document building
def build_document(row: pd.Series) -> Document:
    """Build one document per job: the description as content, job fields as metadata."""

    title = row["job_title"]
    description = row["job_summary"]
    skills = parse_skills(row["job_skills"])
    job_level = row["job level"]
    work_type = row["job_type"].lower()

    metadata = {
        "job_id": job_id_from_link(row["job_link"]),
        "job_link": row["job_link"],
        "title": title,
        "company": row["company"],
        "location": row["job_location"],
        "country": row["search_country"],
        "work_type": work_type,
        "job_level": job_level,
        "seniority": seniority_from_title(title),
        "skills": skills,
    }

    return Document(page_content=description, metadata=metadata)


def load_and_prepare_data(file_path: str) -> list[Document]:
    """Load the CSV, clean and de-duplicate it, and build one Document per job."""
    df = pd.read_csv(file_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    total = len(df)
    df = df[EXPECTED_COLUMNS].copy()
    # Clean every cell in the table
    for column in EXPECTED_COLUMNS:
        df[column] = df[column].map(clean_text)

    # Remove rows with no description
    df = df[df["job_summary"] != ""]
    no_description = total - len(df)

    # Remove duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["job_title", "company", "job_summary"], keep="first"
    )
    duplicates = before_dedup - len(df)

    documents = [build_document(row) for _, row in df.iterrows()]

    logger.info(
        "Loaded %d rows | dropped %d without description | removed %d duplicates | %d jobs kept",
        total,
        no_description,
        duplicates,
        len(documents),
    )

    return documents
