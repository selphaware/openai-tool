import os
import streamlit as st
import openai
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from docx import Document
import fitz  # PyMuPDF

# Configure logging
log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
log_filename = os.path.join(log_folder, f"log_{current_time}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Get OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_KEY')  # Note: Use 'OPENAI_KEY' for the environment variable

# Function to list available models
def list_models():
    try:
        response = openai.Model.list()
        return [model['id'] for model in response['data']]
    except openai.error.OpenAIError as e:
        st.error(f"Error fetching models: {e}")
        return []

# Define a function to handle GPT-4 execution
def execute_gpt4(system_instruction: str, prompt: str, documents: Optional[List[str]] = None, rag_config: Optional[Dict[str, Any]] = None, model_type: str = "gpt-4") -> str:
    try:
        logging.info("Executing GPT-4 with system_instruction: %s, prompt: %s, model_type: %s", system_instruction, prompt, model_type)
        if documents:
            logging.info("Documents provided: %s", documents)
        if rag_config:
            logging.info("RAG configuration: %s", rag_config)

        # Prepare input for GPT-4
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        if documents:
            for doc in documents:
                messages.append({"role": "user", "content": doc})

        # Call the GPT-4 model using the updated API
        response = openai.ChatCompletion.create(
            model=model_type,
            messages=messages,
            max_tokens=1024
        )

        result = response['choices'][0]['message']['content']
        logging.info("GPT-4 response: %s", result)
        return result
    except Exception as e:
        logging.error("Error during GPT-4 execution: %s", e)
        st.error(f"Error: {e}")
        return ""

# Function to read text from DOCX files
def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to read text from PDF files
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Define the main Streamlit app
def main():
    st.title("GPT-4 Document Processor")
    logging.info("Application started")

    # Fetch available models
    available_models = list_models()
    if not available_models:
        st.error("No models available. Please check your API key and network connection.")
        return

    # Model type selection
    model_type = st.selectbox(
        "Select Model Type",
        options=available_models,
        index=available_models.index("gpt-4") if "gpt-4" in available_models else 0,
        help="Choose the model type to use for processing."
    )

    # System instruction input
    system_instruction = st.text_area(
        "System Instruction",
        value="Summarize the following documents:",
        help="Provide the system instruction for GPT-4"
    )

    # Prompt input
    prompt = st.text_area(
        "Prompt",
        help="Provide the prompt for GPT-4"
    )

    # Document upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx"]
    )

    # Display uploaded files
    documents = []
    if uploaded_files:
        for file in uploaded_files:
            if file.type == "text/plain":
                file_text = file.read().decode("utf-8")
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_text = read_docx(file)
            elif file.type == "application/pdf":
                file_text = read_pdf(file)
            else:
                st.error(f"Unsupported file type: {file.type}")
                continue
            documents.append(file_text)
            st.write(f"Uploaded Document: {file.name}")
            logging.info("Uploaded Document: %s", file.name)

    # RAG configuration
    st.sidebar.header("RAG Configuration")
    rag_enabled = st.sidebar.checkbox("Enable RAG", value=False)

    if rag_enabled:
        text_size = st.sidebar.number_input("Text Size (int)", value=512, help="Size of the text chunks to be processed.")
        chunk_size = st.sidebar.number_input("Chunk Size (int)", value=128, help="Number of tokens per chunk.")
        top_k = st.sidebar.number_input("Top K (int)", value=5, help="Number of top documents to retrieve.")
        threshold = st.sidebar.number_input("Threshold (float)", value=0.5, step=0.1, help="Threshold for filtering retrieved documents.")

        rag_settings = {
            "text_size": text_size,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "threshold": threshold
        }
        logging.info("RAG settings enabled: %s", rag_settings)
    else:
        rag_settings = None

    # Execute button
    if st.button("Execute"):
        with st.spinner("Processing..."):
            result = execute_gpt4(system_instruction, prompt, documents, rag_settings, model_type)
            st.success("Execution complete!")
            st.text_area("Response", value=result, height=300)

            # Option to download the result
            st.download_button(
                label="Download Response",
                data=result,
                file_name="response.txt",
                mime="text/plain"
            )
            logging.info("Execution complete and response downloaded")

if __name__ == "__main__":
    main()
