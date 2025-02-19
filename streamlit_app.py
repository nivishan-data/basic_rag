"""
Streamlit application for Document Q&A using Retrieval-Augmented Generation (RAG).
This application provides a chat interface for users to ask questions about their documents
and receive AI-generated responses based on the document content.
"""

# Standard library imports
import asyncio
import logging
from typing import List, Dict, Any

# Third-party imports
import streamlit as st

# Local imports
from rag import primary_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PAGE_TITLE = "Document Q&A Bot"
PAGE_ICON = "🤖"
CHAT_PLACEHOLDER = "Ask a question about your documents..."
THINKING_MESSAGE = "Thinking..."

def run_agent_sync_wrapper(query: str) -> Any:
    """
    Creates a new asyncio event loop to run the agent synchronously.

    Args:
        query (str): The user's question to be processed by the agent.

    Returns:
        Any: The agent's response object.

    Raises:
        Exception: If there's an error during agent execution.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = primary_agent.run_sync(query)
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise

def init_chat_history() -> None:
    """Initialize the chat history in session state if it doesn't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized new chat history")

def display_chat_history() -> None:
    """Display all messages in the chat history with appropriate styling."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str) -> None:
    """
    Process user input and generate agent response.

    Args:
        prompt (str): The user's input question.
    """
    try:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner(THINKING_MESSAGE):
                response = run_agent_sync_wrapper(prompt)
                if response and hasattr(response, 'data'):
                    st.markdown(response.data)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.data}
                    )
                    logger.info("Successfully generated response")
                else:
                    error_message = "I apologize, but I couldn't generate a proper response. Please try again."
                    st.error(error_message)
                    logger.error("Failed to generate valid response")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.error(error_message)
        logger.error(f"Error in handle_user_input: {str(e)}")

def display_sidebar() -> None:
    """Display sidebar with information and controls."""
    with st.sidebar:
        st.title("About")
        st.markdown("""
        ### 🤖 Document Q&A Bot
        
        This intelligent Q&A system uses **Retrieval-Augmented Generation (RAG)** 
        to provide accurate answers based on your documents.
        
        #### How it works:
        1. 🔍 Your question triggers a search through document chunks
        2. 📑 Relevant content is retrieved from the documents
        3. 🧠 An AI model generates a comprehensive answer
        4. ✨ The response is presented with context
        
        #### Tips:
        - Ask specific questions
        - Provide context in your questions
        - Use clear and concise language
        """)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            logger.info("Chat history cleared")
            st.experimental_rerun()

def main() -> None:
    """Main application entry point."""
    try:
        # Configure the Streamlit page
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout="wide"
        )
        
        # Custom CSS for better styling
        st.markdown("""
            <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .stButton button {
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Main title with emoji
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        
        # Initialize and display chat interface
        init_chat_history()
        display_chat_history()
        display_sidebar()
        
        # User input
        prompt = st.text_input("Question", placeholder=CHAT_PLACEHOLDER)
        if prompt:
            handle_user_input(prompt)
            
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()
