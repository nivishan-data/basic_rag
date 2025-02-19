# rag.py

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# ------------------------------------------------------------------------------
# Load environment variables (e.g., REASONING_MODEL_ID and TOOL_MODEL_ID)
load_dotenv()
reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")

# ------------------------------------------------------------------------------
# Helper function: Create a local model instance using Ollama.
def get_local_model(model_id: str):
    return OpenAIModel(model_name=model_id, base_url="http://localhost:11434/v1", api_key="ollama")

# ------------------------------------------------------------------------------
# Create a reasoning agent with a static system prompt and custom model settings.
reasoning_model = get_local_model(reasoning_model_id)
reasoner = Agent(
    reasoning_model,
    system_prompt="Based on the provided context, answer concisely.",
    model_settings={"max_tokens": 1000}  # Adjust as needed.
)

# ------------------------------------------------------------------------------
# Set up the vector store and embeddings for document retrieval.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db2")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# ------------------------------------------------------------------------------
# Create the primary agent for tool-calling using the local model.
tool_model = get_local_model(tool_model_id)
primary_agent = Agent(tool_model)

# ------------------------------------------------------------------------------
# Register the RAG tool using the agent's asynchronous tool decorator.
@primary_agent.tool
async def rag_with_reasoner(ctx: RunContext[None], user_query: str) -> str:
    """
    This asynchronous tool performs a Retrieval-Augmented Generation (RAG) task:
      - It searches the vector store for documents relevant to the query.
      - It concatenates the retrieved document contents into a context.
      - It builds a prompt including both the context and the query.
      - It uses the reasoning agent to generate a concise answer.
    """
    # Retrieve the top 3 similar documents.
    docs = vectordb.similarity_search(user_query, k=3)
    # Combine document contents into one context string.
    context = "\n\n".join(doc.page_content for doc in docs)
    # Construct the prompt.
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
If there isn't sufficient information, provide a better query for RAG.

Context:
{context}

Question: {user_query}

Answer:"""
    # Use the asynchronous run method of the reasoning agent.
    response = await reasoner.run(prompt)
    return response.data

# ------------------------------------------------------------------------------
# Main function: simple command-line interface for testing.
'''
def main():
    print("Document Q&A Bot")
    print("Type your question (or 'exit' to quit):")
    while True:
        user_query = input("Your question: ")
        if user_query.lower() == "exit":
            break
        # Run the primary agent synchronously.
        result = primary_agent.run_sync(user_query)
        print("Answer:", result.data)

if __name__ == "__main__":
    main()
'''