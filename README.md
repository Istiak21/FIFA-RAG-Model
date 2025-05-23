# ⚽ Euro 2020 and FIFA 2002-2022 Question-Answering RAG System

## 📌 Overview

This repository contains two custom Retrieval-Augmented Generation (RAG) question-answering systems built for major international football tournaments:

1. **FIFA World Cup (2002–2022) RAG Model**
2. **UEFA Euro 2020 RAG Model**

Both systems scrape tournament data from Wikipedia, process and chunk the text, embed it into a vector database using `FAISS` and `Hugging Face` sentence embeddings, and then answer natural language questions using the `Together.ai` API and the `meta-llama/Llama-3-8b-chat-hf` model — all powered by LangChain and BeautifulSoup.

---

## 🏆 Projects

### 📖 1️⃣ FIFA World Cup (2002–2022) RAG Model

- Scrapes detailed match reports, tables, squads, and summaries from Wikipedia pages of the FIFA World Cups from 2002 to 2022.
- Splits text into overlapping chunks for context preservation.
- Converts text into vector embeddings using `all-MiniLM-L6-v2`.
- Stores embeddings in a `FAISS` vector database.
- Uses a retriever with **Maximal Marginal Relevance (MMR)** for better context diversity.
- Accepts user questions with optional year-specific filtering (e.g., _"Who won in 2010?"_).
- Generates accurate answers grounded in retrieved context via the `Together.ai` chat API.
- Displays sources alongside answers.

**File:** `FIFA 2002-2022.py`

---

### 📖 2️⃣ UEFA Euro 2020 RAG Model

- Scrapes the Wikipedia page for UEFA Euro 2020.
- Splits article text into overlapping chunks.
- Converts text into vector embeddings using `all-MiniLM-L6-v2`.
- Stores embeddings in a `FAISS` vector database.
- Uses a similarity-based retriever.
- Accepts user questions and provides context-aware answers.
- Uses `Together.ai` and the `meta-llama/Llama-3-8b-chat-hf` model for natural language generation.

**File:** `Euro 2020.py`

---

## 🛠️ Technologies & Libraries

- Python
- LangChain
- Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS Vector Store
- BeautifulSoup (web scraping)
- Together.ai API (for LLM-powered responses)
- TQDM (progress bars)
- dotenv (environment variable management)

---

## 📊 Example Questions

- **Who was the top scorer in the 2010 World Cup?**
- **How many goals were scored in the Euro 2020 final?**
- **Winner in 2018**
- **Which country hosted the 2006 World Cup?**


## 📣 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Together.ai](https://www.together.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Wikipedia](https://www.wikipedia.org/)
- Cardiff NLP for the pretrained `all-MiniLM-L6-v2` embedding model


