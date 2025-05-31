from flask import Flask, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Set environment variables for optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# === Step 1: Load and Split Your Document ===
print("Loading and splitting document...")
loader = TextLoader(".\luis_info.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# === Step 2: Generate Embeddings and Store with FAISS ===
print("Creating embeddings and storing vectors...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding_model)

# === Step 3: Load Quantized Phi-3.5-mini Model ===
print("Loading quantized Phi-3.5-mini model...")
model_path = "./quantized_microsoft_Phi_3.5_mini_instruct"

try:
    # Load the quantized model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    # Create pipeline with SHORT response settings
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,  # Reduced from 256 to 50 tokens (~200 characters)
        min_new_tokens=10,   # Minimum tokens to generate
        temperature=0.3,     # Lower temperature for more focused responses
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Higher penalty to avoid repetition
        return_full_text=False,
        # Additional parameters for shorter responses
        length_penalty=1.5,      # Penalty for longer sequences
        early_stopping=True,     # Stop early when appropriate
        no_repeat_ngram_size=3   # Avoid repeating 3-gram sequences
    )
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ Successfully loaded quantized Phi-3.5-mini model with SHORT response settings")
    
except Exception as e:
    print(f"❌ Error loading quantized model: {e}")
    print("Falling back to Ollama...")
    from langchain.llms import Ollama
    llm = Ollama(
        model="gemma3:4b",
        num_predict=80,  # Limit tokens for Ollama too
        temperature=0.3
    )

# === Step 4: Create Custom Prompt Template for Short Responses ===
short_prompt_template = """Use the following context to answer the question. 
RULES:
1. Always use third person: "Luis is...", "He works as...", "His skills include...", "Alex enjoys..."
2. NEVER provide cooking instructions, recipes, or step-by-step cooking guides
3. If asked about cooking recipes/instructions, say: "Luis enjoys cooking as a hobby, but I don't provide recipes or cooking instructions."
4. Keep responses concise (under 200 words for speed)
5. Only use information from the context about Luis/Alex
6. Refer to him as either "Luis", "Alex", or "Luis Alexander" but always in third person


Context: {context}

Question: {question}

Short Answer:"""

PROMPT = PromptTemplate(
    template=short_prompt_template,
    input_variables=["context", "question"]
)

# === Step 5: Create a RAG Chain with Custom Prompt ===
print("Creating Retrieval-Augmented QA chain with short response settings...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}),  # Reduced to 2 chunks for shorter context
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  # Use custom prompt
)

# === Step 6: Flask API Setup ===
app = Flask(__name__)

def truncate_response(text, max_chars=300):
    """Truncate response to maximum characters"""
    if len(text) <= max_chars:
        return text
    
    # Try to cut at sentence boundary
    sentences = text[:max_chars].split('.')
    if len(sentences) > 1:
        return '.'.join(sentences[:-1]) + '.'
    
    # If no sentence boundary, cut at word boundary
    words = text[:max_chars].split()
    return ' '.join(words[:-1]) + '...'

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        query = data.get('query')
        max_chars = data.get('max_chars', 300)  # Allow custom character limit
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        # Get the response from the RAG model
        print(f"Processing query: {query}")
        result = qa_chain({"query": query})
        
        # Extract and truncate answer
        raw_answer = result["result"]
        answer = truncate_response(raw_answer, max_chars)
        
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "metadata": doc.metadata
                })
        
        response_data = {
            "answer": answer,
            "raw_answer": raw_answer,  # Include full answer for reference
            "answer_length": len(answer),
            "sources": sources,
            "model": "microsoft/Phi-3.5-mini-instruct (quantized - short responses)",
            "query": query,
            "max_chars_requested": max_chars
        }
        
        print(f"Response generated successfully ({len(answer)} chars)")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/ask_short', methods=['POST'])
def ask_question_very_short():
    """Ultra-short responses (under 100 characters)"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        # Modify query to request short answer
        short_query = f"Answer in one sentence: {query}"
        
        print(f"Processing short query: {short_query}")
        result = qa_chain({"query": short_query})
        
        # Extract and heavily truncate answer
        raw_answer = result["result"]
        answer = truncate_response(raw_answer, 100)
        
        response_data = {
            "answer": answer,
            "answer_length": len(answer),
            "model": "microsoft/Phi-3.5-mini-instruct (ultra-short mode)",
            "query": query
        }
        
        print(f"Short response generated ({len(answer)} chars)")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": "quantized Phi-3.5-mini (short responses)" if 'pipe' in globals() else "fallback model",
        "vector_db_chunks": len(chunks),
        "max_tokens": 100,
        "response_mode": "SHORT"
    })

@app.route('/info', methods=['GET'])
def model_info():
    """Get information about the loaded model and documents"""
    return jsonify({
        "model": "microsoft/Phi-3.5-mini-instruct (quantized float16 - short responses)",
        "model_path": model_path,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "document_chunks": len(chunks),
        "chunk_size": 500,
        "chunk_overlap": 50,
        "response_settings": {
            "max_tokens": 100,
            "default_max_chars": 300,
            "temperature": 0.3,
            "length_penalty": 1.5
        }
    })

@app.route('/config', methods=['POST'])
def update_config():
    """Update response length configuration"""
    try:
        data = request.get_json()
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.3)
        
        # Update pipeline configuration
        if 'pipe' in globals():
            pipe.model.generation_config.max_new_tokens = max_tokens
            pipe.model.generation_config.temperature = temperature
            
        return jsonify({
            "status": "Configuration updated",
            "max_tokens": max_tokens,
            "temperature": temperature
        })
        
    except Exception as e:
        return jsonify({"error": f"Configuration update failed: {str(e)}"}), 500

# Start the Flask server
if __name__ == '__main__':
    print("\n🚀 Starting RAG API server with SHORT RESPONSE mode...")
    print("📋 Available endpoints:")
    print("  POST /ask - Ask questions (default: 300 char limit)")
    print("  POST /ask_short - Ask questions (ultra-short: 100 char limit)")
    print("  POST /config - Update response length settings")
    print("  GET /health - Check server health")
    print("  GET /info - Get model information")
    print("\n💡 Example usage:")
    print('  curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"query": "Your question", "max_chars": 200}\'')
    print('  curl -X POST http://localhost:5000/ask_short -H "Content-Type: application/json" -d \'{"query": "Your question"}\'')
    print("\n🌐 Server running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False)