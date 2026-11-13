from src.indexing.preprocess import load_and_prepare_data
from src.indexing.vectorstore import create_vectorstore
import time
import logging

# Configure logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main():
    start = time.time()
    logging.info("Loading documents...")
    documents = load_and_prepare_data(file_path="data/linkedin_job_postings.csv")
    logging.info(f"Loaded {len(documents)} documents.")

    logging.info("Storing in pinecone...")
    create_vectorstore(documents=documents)

    logging.info(f"Completed in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
