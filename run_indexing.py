from src.indexing.preprocess import load_and_prepare_data
from src.indexing.vectorstore import create_vectorstore
import time


def main():
    start = time.time()
    print("Loading documents...")
    documents = load_and_prepare_data(file_path="data/LF_Jobs.csv")
    print(f"Loaded {len(documents)} document.")

    print("Storing in pinecone...")
    create_vectorstore(documents=documents)

    print(f"Completed in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
