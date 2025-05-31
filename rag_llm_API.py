from flask import Flask, request, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re

# === Ultra-Lightweight RAG System for 416MB RAM ===

class UltraLightRAG:
    def __init__(self):
        self.embedding_model = None
        self.chunks = []
        self.embeddings = None
        self.responses = {}
        
    def load_system(self):
        """Load the ultra-lightweight system"""
        try:
            # Load embedding model (~90MB)
            print("Loading embedding model (~90MB)...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load pre-computed embeddings
            if os.path.exists('lightweight_qa.pkl'):
                print("Loading pre-computed embeddings...")
                with open('lightweight_qa.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.embeddings = data['embeddings']
            else:
                print("Creating embeddings from text...")
                self._create_embeddings()
            
            # Load simple response templates
            self._load_response_templates()
            
            print("✅ Ultra-lightweight RAG system loaded!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False
    
    def _create_embeddings(self):
        """Create embeddings from portfolio text"""
        
        # Your portfolio data
        portfolio_data = """
        Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador.
        He specializes in Artificial Intelligence and machine learning.
        Luis works with TensorFlow, PyTorch, and scikit-learn.
        He founded My Software SV, his own software company.
        Luis has 3+ years of professional experience.
        He has completed 15+ projects including SaaS systems, ERP systems, and mobile apps.
        Luis enjoys boxing, tennis, cooking, and technology.
        He studied Software Engineering at Universidad Centroamericana (UCA) for 5 years.
        Luis worked as an intern at Fe y Alegría doing technical support.
        He built a RAG Portfolio Assistant using LangChain and quantized LLMs.
        Luis created a SaaS Billing System with React and REST APIs.
        He developed a Judge Gymnasts App using React Native and Expo.
        Luis built an Enterprise Digital Billing API with Java Spring Boot.
        He created a Multi-Branch ERP System using C# .NET.
        Luis is working on an AI Voice Translation System for anime.
        He contributed to an Open-Source Healthcare AI project with Omdena.
        Luis specializes in computer vision, natural language processing, and audio processing.
        He has experience with AWS, Google Cloud, and DigitalOcean.
        Luis uses Docker, Jenkins, and CI/CD pipelines.
        He knows Python, Java, C++, JavaScript, and C#.
        Luis is learning R for statistical analysis.
        His contact email is alexmtzai2002@gmail.com.
        """
        
        # Split into chunks
        self.chunks = [chunk.strip() for chunk in portfolio_data.split('.') if chunk.strip()]
        
        # Create embeddings
        self.embeddings = self.embedding_model.encode(self.chunks)
        
        # Save for future use
        data = {'chunks': self.chunks, 'embeddings': self.embeddings}
        with open('lightweight_qa.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    def _load_response_templates(self):
        """Load simple response templates for common questions"""
        self.responses = {
            'name': "Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador.",
            'age': "Luis is 23 years old.",
            'profession': "Luis is an AI Engineer and Software Engineer who founded My Software SV.",
            'skills': "Luis specializes in AI/ML with TensorFlow, PyTorch, Python, Java, and cloud technologies.",
            'experience': "Luis has 3+ years of experience and has completed 15+ projects.",
            'education': "Luis studied Software Engineering at Universidad Centroamericana (UCA) for 5 years.",
            'company': "Luis founded and runs My Software SV, his own software company.",
            'projects': "Luis has built SaaS systems, ERP systems, mobile apps, and AI projects.",
            'contact': "You can reach Luis at alexmtzai2002@gmail.com.",
            'interests': "Luis enjoys boxing, tennis, cooking, and technology.",
            'location': "Luis is based in El Salvador."
        }
    
    def find_best_response(self, query):
        """Find best response using simple matching + embeddings"""
        
        query_lower = query.lower()
        
        # First, try simple keyword matching (very fast)
        if any(word in query_lower for word in ['name', 'who is', 'called']):
            return self.responses['name']
        elif any(word in query_lower for word in ['age', 'old', 'years']):
            return self.responses['age']
        elif any(word in query_lower for word in ['job', 'work', 'profession', 'engineer']):
            return self.responses['profession']
        elif any(word in query_lower for word in ['skill', 'technology', 'programming', 'languages']):
            return self.responses['skills']
        elif any(word in query_lower for word in ['experience', 'years', 'worked']):
            return self.responses['experience']
        elif any(word in query_lower for word in ['education', 'study', 'university', 'uca']):
            return self.responses['education']
        elif any(word in query_lower for word in ['company', 'business', 'software sv']):
            return self.responses['company']
        elif any(word in query_lower for word in ['project', 'built', 'created', 'developed']):
            return self.responses['projects']
        elif any(word in query_lower for word in ['contact', 'email', 'reach']):
            return self.responses['contact']
        elif any(word in query_lower for word in ['hobby', 'interest', 'enjoy', 'boxing', 'tennis']):
            return self.responses['interests']
        elif any(word in query_lower for word in ['location', 'where', 'salvador']):
            return self.responses['location']
        
        # If no keyword match, use semantic search (slower but more accurate)
        try:
            query_embedding = self.embedding_model.encode([query])
            similarities = np.dot(query_embedding, self.embeddings.T)[0]
            best_idx = np.argmax(similarities)
            
            if similarities[best_idx] > 0.3:  # Threshold for relevance
                return f"Luis {self.chunks[best_idx]}"
            else:
                return "I don't have specific information about that. You can contact Luis directly at alexmtzai2002@gmail.com for more details."
                
        except Exception as e:
            return "I'm having trouble processing that question right now."

# Initialize the ultra-light RAG system
rag_system = UltraLightRAG()

# Flask app
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        # Get response from ultra-light system
        answer = rag_system.find_best_response(query)
        
        response_data = {
            "answer": answer,
            "query": query,
            "model": "Ultra-Lightweight RAG (416MB RAM optimized)",
            "method": "keyword_matching + embeddings"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/ask_short', methods=['POST'])
def ask_short():
    """Ultra-short responses"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        response = rag_system.find_best_response(query)
        
        # Truncate to first sentence
        short_response = response.split('.')[0] + '.'
        
        return jsonify({
            "answer": short_response,
            "query": query,
            "model": "Ultra-Lightweight (short mode)"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "Ultra-Lightweight RAG",
        "memory_usage": "~150MB",
        "ram_limit": "416MB",
        "optimization": "keyword_matching + sentence_transformers"
    })

@app.route('/memory', methods=['GET'])
def memory_info():
    """Check memory usage"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    return jsonify({
        "memory_usage_mb": round(memory_mb, 1),
        "available_ram_mb": 416,
        "memory_percentage": round((memory_mb / 416) * 100, 1),
        "status": "OK" if memory_mb < 350 else "WARNING"
    })

if __name__ == '__main__':
    print("🚨 Starting Ultra-Lightweight RAG for 416MB RAM")
    print("📊 Expected memory usage: ~150MB")
    
    # Load the system
    if rag_system.load_system():
        print("✅ System loaded successfully!")
        print("🌐 Server starting on http://localhost:5000")
        print("\n📋 Endpoints:")
        print("  POST /ask - Ask questions")
        print("  POST /ask_short - Short responses") 
        print("  GET /health - System health")
        print("  GET /memory - Memory usage")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load system")