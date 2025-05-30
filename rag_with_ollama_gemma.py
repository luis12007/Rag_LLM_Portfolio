# rag_with_ollama_gemma.py

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os
os.environ["OLLAMA_NUM_THREADS"] = "4"

# === Step 1: Load and Split Your Document ===
print("Loading and splitting document...")
loader = TextLoader(".\luis_info.txt")  # <-- Replace with your file
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# === Step 2: Generate Embeddings and Store with FAISS ===
print("Creating embeddings and storing vectors...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding_model)

""" embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index", embedding_model) """


# === Step 3: Load Gemma via Ollama ===
print("Loading Ollama model (gemma3:4b)...")
llm = Ollama(model="gemma3:4b")

# === Step 4: Create a RAG Chain ===
print("Creating Retrieval-Augmented QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    chain_type="stuff"
)

# === Step 5: Ask Questions ===
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break
    response = qa_chain.run(query)
    print(f"\n📌 Answer: {response}")
