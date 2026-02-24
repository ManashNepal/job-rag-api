from src.indexing.preprocess import load_and_prepare_data
from src.indexing.chunking import chunk_documents
from src.indexing.vectorstore import create_vectorstore


def main():
    print("Loading documents...")
    documents = load_and_prepare_data(file_path="data/LF_Jobs.csv")
    print(f"Loaded {len(documents)} document.")

    print("Chunking documents...")
    chunked_documents = chunk_documents(documents=documents)
    print(f"Created {len(chunked_documents)} chunks.")

    print("Storing in pinecone...")
    create_vectorstore(documents=chunked_documents)


if __name__ == "__main__":
    main()
