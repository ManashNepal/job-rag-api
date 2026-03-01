import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from langchain_classic.schema import Document


def clean_html(raw_html: str) -> str:
    """Remove HTML tags and return clean text."""

    # Check if description is empty
    if pd.isna(raw_html):
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    return " ".join(text.split())


def safe_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_document(row: pd.Series) -> Document:
    """Build formatted documents for each row."""

    job_description = clean_html(row["Job Description"])
    # Avoid hallucinated response
    if len(job_description.strip()) < 50:
        job_description = "No detailed description provided."

    title = safe_value(row["Job Title"])
    company = safe_value(row["Company Name"])
    location = safe_value(row["Job Location"])
    level = safe_value(row["Job Level"])
    category = safe_value(row["Job Category"])

    formatted_text = f"""
---
Job Title: {title}
Company: {company}
Location: {location}
Level: {level}
Category: {category}

Description:
{job_description}
---
    """

    metadata = {
        "id": safe_value(row["ID"]),
        "title": title,
        "company": company,
        "location": location,
        "level": level,
        "category": category,
        "publication_date": safe_value(row["Publication Date"]),
    }

    return Document(page_content=formatted_text, metadata=metadata)


def load_and_prepare_data(file_path: str) -> List[Document]:
    """Load CSV file and prepare document for indexing."""

    df = pd.read_csv(file_path)
    documents = []

    for _, row in df.iterrows():
        documents.append(build_document(row))

    return documents
