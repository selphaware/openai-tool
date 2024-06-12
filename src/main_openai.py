import os
import streamlit as st
import openai
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from docx import Document
import fitz  # PyMuPDF
import pandas as pd

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

# Get Azure OpenAI API credentials from environment variables
api_type = os.getenv('AZURE_OPENAI_API_TYPE', 'azure')
api_base = os.getenv('AZURE_OPENAI_API_BASE')
api_version = os.getenv('AZURE_OPENAI_API_VERSION')
api_key = os.getenv('AZURE_OPENAI_API_KEY')

# Setup OpenAI configuration for Azure
openai.api_type = api_type
openai.api_base = api_base
openai.api_version = api_version
openai.api_key = api_key

# Function to list available models
def list_models():
    try:
        response = openai.Model.list()
        return [model['id'] for model in response['data']]
    except openai.error.OpenAIError as e:
        st.error(f"Error fetching models: {e}")
        return []

# Function to chunk text into smaller parts
def chunk_text(text: str, max_tokens: int) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Define a function to handle GPT-4 execution
def execute_gpt4(system_instruction: str, prompt: str, documents: Optional[List[str]] = None, rag_config: Optional[Dict[str, Any]] = None, engine: str = "gpt-4-32k", temperature: float = 0.7) -> str:
    try:
        logging.info("Executing GPT-4 with system_instruction: %s, prompt: %s, engine: %s, temperature: %s", system_instruction, prompt, engine, temperature)
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
                chunks = chunk_text(doc, rag_config['chunk_size'])
                for chunk in chunks:
                    messages.append({"role": "user", "content": chunk})
            logging.info("Messages prepared for API call: %s", messages)

        # Ensure total tokens do not exceed the model's limit
        total_message_tokens = sum(len(m['content']) for m in messages)
        if total_message_tokens + rag_config['text_size'] > 32768:
            raise ValueError("Combined length of input messages and completion exceeds the model's maximum context length.")

        # Call the GPT-4 model using the updated API
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=messages,
            max_tokens=rag_config['text_size'],
            temperature=temperature
        )

        result = response['choices'][0]['message']['content']
        logging.info("GPT-4 response: %s", result)
        return result
    except openai.error.OpenAIError as e:
        logging.error("Error during GPT-4 execution: %s", e)
        st.error(f"Error: {e}")
        return ""
    except ValueError as e:
        logging.error("Token limit error: %s", e)
        st.error(f"Error: {e}")
        return ""

# Function to read text from DOCX files
def read_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    logging.info("Read DOCX content")
    return text

# Function to read text from PDF files
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    logging.info("Read PDF content")
    return text

# Function to read text from CSV files
def read_csv(file):
    df = pd.read_csv(file)
    text = df.to_string(index=False)
    logging.info("Read CSV content")
    return text

# Function to read text from Excel files
def read_excel(file):
    df = pd.read_excel(file)
    text = df.to_string(index=False)
    logging.info("Read Excel content")
    return text

# Function to read text from Feather files
def read_feather(file):
    df = pd.read_feather(file)
    text = df.to_string(index=False)
    logging.info("Read Feather content")
    return text

# Function to read text from Parquet files
def read_parquet(file):
    df = pd.read_parquet(file)
    text = df.to_string(index=False)
    logging.info("Read Parquet content")
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
        index=available_models.index("gpt-4-32k") if "gpt-4-32k" in available_models else 0,
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
        type=["txt", "pdf", "docx", "csv", "xlsx", "feather", "parquet"]
    )

    # Display uploaded files
    documents = []
    if uploaded_files:
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1].lower()
            try:
                if file_extension == ".txt":
                    file_text = file.read().decode("utf-8")
                elif file_extension == ".docx":
                    file_text = read_docx(file)
                elif file_extension == ".pdf":
                    file_text = read_pdf(file)
                elif file_extension == ".csv":
                    file_text = read_csv(file)
                elif file_extension == ".xlsx":
                    file_text = read_excel(file)
                elif file_extension == ".feather":
                    file_text = read_feather(file)
                elif file_extension == ".parquet":
                    file_text = read_parquet(file)
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    continue
                documents.append(file_text)
                st.write(f"Uploaded Document: {file.name}")
                logging.info("Uploaded Document: %s", file.name)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                logging.error(f"Error reading {file.name}: {e}")

    # Combine document content if necessary
    if documents:
        combined_content = "\n\n".join(documents)
        documents = [combined_content]  # Use the combined content for processing

    # RAG configuration
    st.sidebar.header("RAG Configuration")
    rag_enabled = st.sidebar.checkbox("Enable RAG", value=False)

    if rag_enabled:
        text_size = st.sidebar.number_input("Text Size (int)", value=10000, help="Size of the text chunks to be processed.")
        chunk_size = st.sidebar.number_input("Chunk Size (int)", value=512, help="Number of tokens per chunk.")
        top_k = st.sidebar.number_input("Top K (int)", value=5, help="Number of top documents to retrieve.")
        threshold = st.sidebar.number_input("Threshold (float)", value=0.5, step=0.1, help="Threshold for filtering retrieved documents.")
        temperature = st.sidebar.slider("Temperature (float)", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Control the randomness of the output")

        rag_settings = {
            "text_size": text_size,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "threshold": threshold,
            "temperature": temperature
        }
        logging.info("RAG settings enabled: %s", rag_settings)
    else:
        rag_settings = None

    # Execute button
    if st.button("Execute"):
        with st.spinner("Processing..."):
            result = execute_gpt4(system_instruction, prompt, documents, rag_settings, model_type, rag_settings['temperature'] if rag_settings else 0.7)
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

