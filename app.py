import gradio as gr
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings

# Set your Groq API Key
GROQ_API_KEY = "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"
MODEL_ID = "deepseek-r1-distill-llama-70b"

# Custom Embedding Class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Extract text from PDFs
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        with pdfplumber.open(file.name) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    return text

# Chunk text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Create FAISS index
def create_faiss_index(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings_model = SentenceTransformerEmbeddings()
    return FAISS.from_documents(documents, embeddings_model)

# Process PDFs
def process_pdfs(files):
    text = extract_text_from_pdfs(files)
    chunks = chunk_text(text)
    index = create_faiss_index(chunks)
    return index

# Query System
def query_sop(question, index):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_ID)
    retriever = index.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
    You are an expert in SOP analysis. Answer the user's question using the provided context.
    
    <context>
    {context}
    </context>
    
    **Question**: {input}
    
    **Reasoning**:
    - Step 1: Understand the question
    - Step 2: Extract relevant information
    - Step 3: Provide a structured response
    
    **Answer**:
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

# Gradio Interface
def chatbot_interface(files, question):
    if not files:
        return "Please upload at least one PDF."
    
    index = process_pdfs(files)
    return query_sop(question, index)

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Files(file_types=[".pdf"], label="Upload SOP PDFs"), gr.Textbox(label="Enter your question")],
    outputs="text",
    title="SOP Thinking Chatbot",
    description="Upload SOP PDFs and ask questions."
)

if __name__ == "__main__":
    iface.launch()
