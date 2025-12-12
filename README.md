# 📘 NVIDIA NIM RAG – Document Q&A App

A Retrieval-Augmented Generation (RAG) application built using NVIDIA NIM, LangChain, FAISS, and Streamlit.
Users can upload multiple PDF documents, generate embeddings using NVIDIA Embeddings, and ask questions based only on the document content.

---
## Live APP

You can use deployed version here:
**[Live App](https://nvidia-nim-rag-pdf-qna-app-jsfcyx5rv5oyagirkrnjqz.streamlit.app/)**
---
## 🚀 Features

- 🔐 User enters their own NVIDIA API Key (secure, no key stored in repository)
- 📄 Upload multiple PDF files
- 🔍 Automatic text extraction & chunking
- 🧠 NVIDIA Embeddings for vectorization
- 📚 FAISS Vector Store for fast semantic search
- 💬 Llama-3.3-70B-Instruct model used for answering queries
- ⚡ Clean Streamlit UI
- 🧩 Fully local vector store (no backend required)

---
## 🛠️ Tech Stack
```
| Component  | Technology                               |
| ---------- | ---------------------------------------- |
| LLM        | NVIDIA NIM `meta/llama-3.3-70b-instruct` |
| Embeddings | `NVIDIAEmbeddings`                       |
| Framework  | LangChain                                |
| Vector DB  | FAISS                                    |
| UI         | Streamlit                                |
| PDF Loader | PyPDFLoader                              |
```
---
## 📦 Installation
1. Clone this repository:
```bash
git clone https://github.com/sush-sp777/NVIDIA-NIM-RAG-PDF-Qna-App.git
cd NVIDIA-NIM-RAG-PDF-QnA-App
```
2. Create & activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
🔑 NVIDIA API Key
- This project does not include any API keys in the repo.
- Users must enter their own key inside the application:
  
1. Go to: https://build.nvidia.com/explore/discover
2. Create an account
3. Generate an API key
4. Paste it into the app input box at runtime
▶️ Run the App:
```bash
streamlit run final_app.py
```
---
### 📚 How It Works
1️⃣ Upload PDF Files
Users upload one or more PDF documents.

2️⃣ Embeddings Creation
PDFs → Text → Chunks
NVIDIAEmbeddings converts chunks into embeddings
FAISS stores embeddings locally

3️⃣ Ask a Question
RAG pipeline:
```sql
Retriever → Prompt → LLM → Final Answer
```
The model answers only using context from uploaded documents.
