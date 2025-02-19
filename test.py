import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

client = chromadb.PersistentClient(path="chroma_db2")
collections = client.list_collections()
#print("Collections:", collections)
'''
collection = client.get_collection("langchain")
num_records = collection.count()
print(f"Number of records in 'your_collection': {num_records}")
if num_records > 0:
    data = collection.get()  # Retrieves all records
    print("Data:", data)
'''
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db2")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

user_query = "can you summarize"
docs = vectordb.similarity_search(user_query, k=3)
context = "\n\n".join(doc.page_content for doc in docs)
print("Retrieved documents:", context)