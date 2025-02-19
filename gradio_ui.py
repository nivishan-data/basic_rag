# gradio_ui.py

import gradio as gr
from rag import primary_agent

def ask_question(query: str) -> str:
    # Use run_sync to process the query and return the answer.
    result = primary_agent.run_sync(query)
    return result.data

# Create a Gradio Interface
iface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="Document Q&A Bot",
    description="Ask questions about your documents using Retrieval-Augmented Generation (RAG)."
)

if __name__ == "__main__":
    iface.launch()
