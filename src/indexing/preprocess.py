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
    text = soup.get_text(separator="\n")

    return " ".join(text.split())


def build_document(row: pd.Series) -> Document:
    """Build formatted documents for each row."""

    job_description = clean_html(row["Job Description"])
    tags = row["Tags"] if not pd.isna(row["Tags"]) else ""

    formatted_text = (
        f"Job Title: {row['Job Title']}\n"
        f"Company: {row['Company Name']}\n"
        f"Location: {row['Job Location']}\n"
        f"Level: {row['Job Level']}\n"
        f"Category: {row['Job Category']}\n"
        f"Tags: {tags}\n\n"
        f"Description:\n{job_description}"
    )

    metadata = {
        "id": row["ID"],
        "title": row["Job Title"],
        "company": row["Company Name"],
        "location": row["Job Location"],
        "level": row["Job Level"],
        "category": row["Job Category"],
        "publication_date": row["Publication Date"],
    }

    return Document(page_content=formatted_text, metadata=metadata)


def load_and_prepare_data(file_path: str) -> List[Document]:
    """Load CSV file and prepare document for indexing."""

    df = pd.read_csv(file_path)
    documents = []

    for _, row in df.iterrows():
        documents.append(build_document(row))

    return documents
