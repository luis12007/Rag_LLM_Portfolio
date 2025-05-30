# 🤖 RAG-Powered Personal Portfolio Assistant

A Retrieval-Augmented Generation (RAG) system that provides intelligent responses about Luis Alexander Hernandez Martinez using a quantized Phi-3.5-mini model.

## 🎯 Overview

This project creates a personal AI assistant that can answer questions about Luis's professional background, skills, experience, and interests. The system uses RAG to combine document retrieval with language model generation for accurate, contextual responses.

## ✨ Features

- **RAG Implementation**: Combines FAISS vector search with LLM generation
- **Quantized Models**: Optimized Phi-3.5-mini for faster CPU inference
- **Third-Person Responses**: Always refers to Luis in third person
- **Content Filtering**: Blocks inappropriate requests (cooking recipes, etc.)
- **Multiple APIs**: Standard and ultra-fast response options
- **Model Optimization**: Multiple quantization levels for different speed/quality trade-offs

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-portfolio-assistant.git
cd rag-portfolio-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download and quantize a model** (optional - for better performance)
```bash
python quantizer.py
# Choose option 1 for Phi-3.5-mini
```

4. **Start the API server**
```bash
python rag_llm_API.py
```

5. **Test the assistant**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Luis do?"}'
```

## 📁 Project Structure

```
📦 rag-portfolio-assistant/
├── 📄 rag_llm_API.py          # Main RAG API server
├── 📄 fast_rag_api.py         # Ultra-fast optimized version
├── 📄 quantizer.py            # Model quantization utility
├── 📄 cpu_quantizer.py        # CPU-specific quantization
├── 📄 simple_quantizer.py     # Simple model optimization
├── 📄 luis_info.txt           # Full portfolio information
├── 📄 luis_info_short.txt     # Condensed version for speed
├── 📄 requirements.txt        # Python dependencies
├── 📄 .gitignore             # Git ignore rules
└── 📂 quantized_models/       # Generated quantized models (not in git)
```

## 🔧 API Endpoints

### Main API (`rag_llm_API.py`)

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/ask` | POST | Ask questions about Luis |
| `/health` | GET | Check API status |
| `/benchmark` | GET | Performance test |

### Fast API (`fast_rag_api.py`)

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/ask` | POST | Regular fast processing |
| `/quick` | POST | Ultra-fast pre-computed responses |
| `/health` | GET | Health check |

## 💡 Usage Examples

### Basic Questions
```bash
# Professional background
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Luis do?"}'

# Skills and expertise
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are his technical skills?"}'

# Personal interests
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Luis interests?"}'
```

### Quick Responses
```bash
# Ultra-fast pre-computed answers
curl -X POST http://localhost:5000/quick \
  -H "Content-Type: application/json" \
  -d '{"query": "who is luis"}'
```

## ⚙️ Configuration Options

### Model Paths
Update the model path in the API files:
```python
model_path = "./quantized_microsoft_Phi_3.5_mini_instruct"  # Standard
model_path = "./ultra_fast_microsoft_Phi_3.5_mini_instruct"  # Optimized
model_path = "./super_fast_phi35"  # Ultra-fast
```

### Response Tuning
Adjust generation parameters for speed vs quality:
```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=40,    # Shorter = faster
    temperature=0.3,      # Lower = more focused
    do_sample=True,       # False = faster greedy decoding
)
```

### Content Filtering
Customize blocked topics:
```python
cooking_instruction_keywords = [
    'how to cook', 'recipe for', 'cooking instructions'
    # Add more blocked terms
]
```

## 🔄 Model Optimization

### Quantization Levels

1. **Ultra-Fast** (`simple_quantizer.py`)
   - Float16 + torch.compile
   - 2-3x speed improvement
   - Best for quick responses

2. **Aggressive** (`cpu_quantizer.py`)
   - Float16 + dynamic quantization
   - 3-5x speed improvement
   - Good balance of speed/quality

3. **Standard** (`quantizer.py`)
   - 4-bit quantization
   - Maximum compression
   - Requires bitsandbytes

## 📊 Performance Optimization

### Speed Improvements
- **Document chunking**: Smaller chunks (200 chars) for faster retrieval
- **Reduced context**: Only 1-2 retrieved chunks instead of 3-5
- **Short responses**: 25-40 tokens instead of 256
- **Model caching**: Load once, reuse for all requests
- **Pre-computed answers**: Instant responses for common questions

### Memory Optimization
- **Quantized models**: 4-bit and 8-bit quantization
- **CPU-optimized**: Designed for CPU inference
- **Low memory usage**: Efficient loading strategies

## 🛡️ Content Safety

### Allowed Topics
- ✅ Luis's professional background
- ✅ Technical skills and experience
- ✅ Education and career
- ✅ Personal interests (general)
- ✅ Company information (My Software SV)

### Blocked Content
- ❌ Cooking instructions and recipes
- ❌ Step-by-step how-to guides
- ❌ Personal/private information
- ❌ Non-Luis related queries

## 🔧 Troubleshooting

### Common Issues

**Slow responses (2+ minutes)**
```bash
# Use the fast API
python fast_rag_api.py

# Or optimize the model
python simple_quantizer.py
```

**Model loading errors**
```bash
# Check model path exists
ls ./quantized_microsoft_Phi_3.5_mini_instruct/

# Use fallback model
# The API will automatically fall back to Ollama if local model fails
```

**Memory issues**
```bash
# Use smaller model or reduce max_new_tokens
max_new_tokens=25  # Instead of 50+
```

**bitsandbytes errors**
```bash
# Use CPU quantizer instead
python cpu_quantizer.py
```

## 📋 Requirements

### Core Dependencies
```
torch>=2.0.0
transformers>=4.30.0
langchain>=0.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
flask>=2.3.0
```

### Optional (for quantization)
```
bitsandbytes>=0.41.0
accelerate>=0.20.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 About Luis Alexander

Luis Alexander Hernandez Martinez is a 23-year-old AI Engineer from El Salvador and the founder & CEO of My Software SV. He specializes in machine learning, software development, and has built 15+ software solutions for 20+ clients.

**Contact:**
- 📧 Email: alexmtzai2002@gmail.com
- 🌐 Portfolio: https://portfolio-production-319e.up.railway.app
- 💼 Company: My Software SV

## 🏆 Achievements

- ✅ Built RAG system with quantized LLM
- ✅ 3-5x speed optimization through quantization
- ✅ Intelligent content filtering
- ✅ Multiple API configurations for different use cases
- ✅ CPU-optimized inference pipeline

---

**⚡ Performance:** ~5-15 second response times | **🔒 Security:** Content-filtered responses | **🚀 Optimized:** Multiple quantization levels available