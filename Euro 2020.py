# # Euro 2020 RAG Model

import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from together import Together

# Load environment variable
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

# Scrape EURO 2020 Wikipedia page
def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([para.get_text() for para in paragraphs])
    return text

print("Loading and processing UEFA Euro 2020 data...")
wiki_url = "https://en.wikipedia.org/wiki/UEFA_Euro_2020"
raw_text = scrape_wikipedia(wiki_url)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
texts = text_splitter.split_text(raw_text)

# Convert to Document objects
documents = [Document(page_content=chunk) for chunk in texts]

# Create vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

# Setup retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Together client
client = Together(api_key=api_key)

def format_prompt(query, relevant_docs):
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""You are a football expert assistant specializing in UEFA Euro 2020.
    Always answer based on the provided context. If you don't know, say you don't know.

    Context:
    {context}

    Question: {query}

    Answer:"""
    return prompt

def get_answer(query):
    relevant_docs = retriever.invoke(query)
    prompt = format_prompt(query, relevant_docs)

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[
                {"role": "system", "content": "You are a helpful football expert assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response from Together.ai: {e}")
        return "\nHere are the relevant documents I found:\n" + "\n\n".join([doc.page_content for doc in relevant_docs])

# Interactive question loop
print("\nUEFA Euro 2020 Question Answering System")
print("Type 'exit' or 'quit' to end the session\n")

while True:
    query = input("\nWhat would you like to know about UEFA Euro 2020? ")

    if query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    if not query.strip():
        print("Please enter a valid question.")
        continue

    print("\nThinking...\n")
    answer = get_answer(query)
    print(answer)

