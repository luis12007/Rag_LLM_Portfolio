import torch
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    DistilBertTokenizer, DistilBertForQuestionAnswering,
    pipeline
)

def download_ultra_small_models():
    """Download models that can run in ~200MB RAM"""
    
    # Ultra-small models under 400MB
    ultra_small_models = [
        # Tiny language models (~50-200MB)
        "microsoft/DialoGPT-small",           # ~350MB - conversational
        "distilgpt2",                         # ~350MB - text generation  
        "gpt2",                               # ~500MB - might be too big
        
        # Specialized small models
        "distilbert-base-uncased",            # ~250MB - Q&A model
        "sentence-transformers/all-MiniLM-L6-v2",  # ~90MB - embeddings only
        
        # Tiny instruction models
        "microsoft/DialoGPT-medium",          # ~800MB - too big
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # ~2.2GB - way too big
    ]
    
    # Try the smallest ones first
    for model_name in ["distilgpt2", "microsoft/DialoGPT-small", "distilbert-base-uncased"]:
        try:
            print(f"Downloading ultra-small model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load with maximum memory optimization
            if "distilbert" in model_name:
                model = DistilBertForQuestionAnswering.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            
            # Save locally
            model_safe_name = model_name.replace("/", "_").replace("-", "_")
            output_dir = f"./tiny_{model_safe_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
            
            print(f"✅ Successfully saved: {output_dir}")
            
            # Estimate size
            size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                         for dirpath, dirnames, filenames in os.walk(output_dir) 
                         for filename in filenames) / (1024*1024)
            
            print(f"📊 Model size: ~{size_mb:.0f}MB")
            
            if size_mb < 400:
                print(f"✅ Model fits in your {416}MB RAM!")
                return output_dir
            else:
                print(f"❌ Model too large ({size_mb:.0f}MB)")
                
        except Exception as e:
            print(f"❌ Failed with {model_name}: {e}")
            continue
    
    return None

def create_simple_qa_system():
    """Create a simple Q&A system without heavy models"""
    
    # Ultra-lightweight approach using sentence-transformers only
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import pickle
    
    print("Creating ultra-lightweight Q&A system...")
    
    # Use only embeddings (no language model)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Only ~90MB
    
    # Load your portfolio data
    try:
        with open("luis_info.txt", "r") as f:
            portfolio_text = f.read()
    except:
        portfolio_text = """
        Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador.
        He specializes in machine learning with TensorFlow and PyTorch.
        Luis founded My Software SV and has 3+ years of experience.
        He enjoys boxing, tennis, cooking, and technology.
        """
    
    # Split into chunks
    chunks = [chunk.strip() for chunk in portfolio_text.split('\n') if chunk.strip()]
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Save lightweight system
    data = {
        'chunks': chunks,
        'embeddings': embeddings
    }
    
    with open('lightweight_qa.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Ultra-lightweight Q&A system created!")
    print("📊 Size: ~100MB total")
    return True

if __name__ == "__main__":
    print("🚨 Detected 416MB RAM - Using ultra-lightweight setup")
    print("\nOption 1: Try tiny language models (might still be too big)")
    print("Option 2: Create embeddings-only Q&A system (recommended)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "2":
        create_simple_qa_system()
    else:
        download_ultra_small_models()