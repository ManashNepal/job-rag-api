import logging
import time

from src.indexing.chunking import chunk_documents
from src.indexing.preprocess import load_and_prepare_data
from src.indexing.vectorstore import index_chunks

# Configure logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main():
    start = time.time()

    logging.info("Loading documents...")
    documents = load_and_prepare_data(file_path="data/linkedin_job_postings.csv")

    logging.info("Chunking documents...")
    chunks = chunk_documents(documents)
    logging.info(f"Created {len(chunks)} chunks from {len(documents)} jobs.")

    logging.info("Storing in Qdrant...")
    index_chunks(chunks)

    logging.info(f"Completed in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
