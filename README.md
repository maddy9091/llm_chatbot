# SOP Thinking Chatbot

## Overview
The **SOP Thinking Chatbot** is an AI-powered chatbot that helps users analyze and retrieve information from Standard Operating Procedures (SOPs) in PDF format. Users can upload PDFs, ask questions, and receive structured responses based on the extracted content.

## Features
- **PDF Extraction**: Extracts text from uploaded PDF documents.
- **Text Chunking**: Splits extracted text into manageable chunks for better retrieval.
- **Vector Storage**: Uses FAISS for efficient document embedding and retrieval.
- **LLM-Powered QA**: Utilizes Groq's DeepSeek-R1-Distill-LLama-70B model for intelligent responses.
- **Interactive Interface**: Built with Gradio for easy user interaction.

## Technologies Used
- **Python**
- **Gradio**: For building the web-based chatbot interface.
- **pdfplumber**: For extracting text from PDF files.
- **LangChain**: For text processing, chunking, and retrieval.
- **Sentence Transformers**: For embedding text using `all-MiniLM-L6-v2`.
- **FAISS**: For efficient similarity search and retrieval.
- **Groq API**: For LLM-based question answering.

## Installation
### Prerequisites
Ensure you have Python 3.8 or later installed.

### Steps
1. **Clone the Repository**
   ```sh
   git clone <repository_url>
   cd sop-thinking-chatbot
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set Up API Key**
   - Replace `GROQ_API_KEY` in `app.py` with your valid Groq API key.

## Usage
1. Run the chatbot:
   ```sh
   python app.py
   ```
2. Open the interface in your browser (default: `http://127.0.0.1:7860`).
3. Upload one or more SOP PDFs.
4. Enter a question related to the SOP content.
5. Get structured, AI-generated answers.

## File Structure
```
├── app.py                 # Main chatbot script
├── requirements.txt       # Required Python dependencies
├── README.md              # Documentation
├── data/                  # Folder for sample PDFs (if needed)
```

## Future Enhancements
- Add multi-modal support (images, tables from PDFs)
- Enhance retrieval with hybrid search (dense + sparse)
- Improve UI for better user experience

## License
This project is licensed under the MIT License.

