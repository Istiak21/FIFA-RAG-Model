# # FIFA World Cup context based RAG model


import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from together import Together
from tqdm import tqdm  # For progress bars

# Loading environment variable

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

# List of World Cup Wikipedia pages to scrape
WORLD_CUP_URLS = {
    "2002": "https://en.wikipedia.org/wiki/2002_FIFA_World_Cup",
    "2006": "https://en.wikipedia.org/wiki/2006_FIFA_World_Cup",
    "2010": "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup",
    "2014": "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup",
    "2018": "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup",
    "2022": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup"
}

def scrape_wikipedia(url):
    """Enhanced Wikipedia scraper that gets main content and info tables"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Get all paragraph text
    paragraphs = soup.find_all("p")
    text = "\n".join([para.get_text() for para in paragraphs])

    # Add information from tables (like results, squads, etc.)
    tables = soup.find_all("table", {"class": "wikitable"})
    for table in tables:
        rows = table.find_all("tr")
        table_data = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            table_data.append(" | ".join(cell.get_text(strip=True) for cell in cells))
        text += "\n\nTABLE:\n" + "\n".join(table_data)

    return text

def load_and_process_documents():
    """Load all World Cup data and process into documents"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    print("Loading World Cup data from Wikipedia...")
    for year, url in tqdm(WORLD_CUP_URLS.items()):
        try:
            text = scrape_wikipedia(url)
            # Add year metadata to each chunk
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"year": year, "source": url}
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error processing {year} World Cup: {e}")

    return documents

# Initialize components
print("Initializing system...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = load_and_process_documents()

print(f"\nCreating vector database from {len(documents)} documents...")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance for better diversity
    search_kwargs={"k": 5, "filter": None}
)

client = Together(api_key=api_key)

def format_prompt(query, relevant_docs):
    """Improved prompt formatting with source information"""
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'unknown source')
        year = doc.metadata.get('year', 'unknown year')
        context_parts.append(f"From {year} World Cup ({source}):\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a football expert assistant specializing in FIFA World Cups (2002-2022).
Answer the question using ONLY the provided context. If unsure, say you don't know.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

def get_answer(query, year_filter=None):
    """Get answer with optional year filtering"""
    search_kwargs = {"k": 5}
    if year_filter:
        search_kwargs["filter"] = {"year": year_filter}

    relevant_docs = retriever.invoke(query, search_kwargs=search_kwargs)
    prompt = format_prompt(query, relevant_docs)

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[
                {"role": "system", "content": "You are a precise football historian."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3  # Lower temperature for more factual answers
        )
        answer = response.choices[0].message.content

        # Add sources
        sources = list({doc.metadata.get('source') for doc in relevant_docs})
        return f"{answer}\n\nSources:\n" + "\n".join(sources)
    except Exception as e:
        print(f"Error: {e}")
        return "I couldn't process that request. Here are relevant documents:\n" + \
               "\n\n".join([f"From {doc.metadata.get('year')}:\n{doc.page_content}" 
                          for doc in relevant_docs])

# Interactive interface
print("\nFIFA World Cup (2002-2022) Question Answering System")
print("You can ask about any World Cup between 2002-2022")
print("Add 'in [year]' to filter questions (e.g., 'winner in 2010')")
print("Type 'exit' to quit\n")

while True:
    query = input("\nYour question about World Cups: ").strip()

    if query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    if not query:
        print("Please enter a question.")
        continue

    # Check for year filter in query
    year_filter = None
    if " in " in query.lower():
        query_parts = query.split(" in ")
        query = query_parts[0].strip()
        year_part = query_parts[1].strip()
        if year_part.isdigit() and year_part in WORLD_CUP_URLS:
            year_filter = year_part
            print(f"Filtering for {year_filter} World Cup...")

    print("\nSearching across World Cup archives...\n")
    answer = get_answer(query, year_filter)
    print(answer)


