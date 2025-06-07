# 🤖 AI Chatbot with RAG (Retrieval-Augmented Generation)

This is an intelligent chatbot that leverages **RAG (Retrieval-Augmented Generation)** to provide accurate, up-to-date, and context-aware responses by combining the power of **local LLMs** with **retrieved knowledge from custom documents**. It is designed for enhanced conversation quality in domain-specific applications like customer support, research, and internal knowledge bases.

## 🔍 What is RAG?

RAG combines two powerful components:

1. **Retriever**: Fetches relevant documents or information chunks from a knowledge base (like PDFs, text files, or datasets).
2. **Generator**: Uses a language model (e.g., Llama3 via Ollama) to generate responses augmented with retrieved context.

---

## 🚀 Features

* 📚 Document-aware Q\&A using Retrieval-Augmented Generation
* 💡 Contextual, high-quality responses using local LLMs
* 🗂️ Custom knowledge base support (PDFs, text files, etc.)
* ⚡ Fast and private: everything runs locally
* 🔌 Modular design for easy customization

---

## 🛠️ Tech Stack

* [Python](https://www.python.org/)
* [AutoGen](https://github.com/microsoft/autogen) 
* ChromaDB for vector storage
* [Ollama](https://ollama.com/) for running local LLMs (e.g., Llama3)
* [PyMuPDF / pdfminer.six](https://pypi.org/project/pdfminer.six/) for PDF processing
* \[SentenceTransformers / OpenAI embeddings] for text embedding

---

## 📂 Project Structure

```
ai-chatbot-rag/
├── chatbot.py                 # Main script
├── retriever/
│   ├── document_loader.py     # Load and chunk documents
│   └── vector_store.py        # FAISS/Chroma vector DB logic
├── generator/
│   └── llm_interface.py       # Call to local LLM (Ollama)
├── data/
│   └── knowledge_base/        # Custom documents (PDFs, .txt, etc.)
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Ollama with Llama3**

   ```bash
   ollama run llama3
   ```

3. **Add your documents**
   Place your knowledge base files (PDFs or text) in the `data/knowledge_base/` folder.

4. **Run the chatbot**

   ```bash
   python chatbot.py
   ```

---

## 💡 Use Cases

* 🤖 AI assistant for company documentation or SOPs
* 🏥 Healthcare chatbot with domain-specific content
* 🎓 Educational tutor using academic materials
* 🏛️ Legal or financial assistant using policy documents

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/601b4894-d27c-439b-9697-afc7d9fe88f5)


---

## 🧠 How It Works (RAG Pipeline)

```mermaid
graph TD
    A[User Query] --> B[Retriever (Chroma)]
    B --> C[Relevant Docs]
    C --> D[LLM (Ollama - Llama3)]
    D --> E[Response with Augmented Context]
```

---

## 🙌 Acknowledgements

* Built using inspiration from LangChain and Microsoft AutoGen
* Thanks to open-source contributors in LLM and RAG communities

---

