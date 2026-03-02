from src.rag.service import RAGService

service = RAGService()

query = "python developer remote"

response = service.query(query)

print(response)
