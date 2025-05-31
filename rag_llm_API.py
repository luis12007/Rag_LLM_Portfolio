from flask import Flask, request, jsonify
from flask_cors import CORS
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
        
        # Your comprehensive portfolio data
        portfolio_data = """
        Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador.
        He specializes in Artificial Intelligence and machine learning.
        Luis works with TensorFlow, PyTorch, and scikit-learn for AI development.
        He founded My Software SV, his own software company in El Salvador.
        Luis has 3+ years of professional experience in software development.
        He has completed 15+ projects including SaaS systems, ERP systems, and mobile apps.
        Luis enjoys boxing, tennis, cooking, and technology as hobbies.
        He studied Software Engineering at Universidad Centroamericana (UCA) for 5 years.
        Luis worked as an intern at Fe y Alegría doing technical support from August 2021 to January 2022.
        He built a RAG Portfolio Assistant using LangChain and quantized LLMs deployed on DigitalOcean.
        Luis created a SaaS Billing System with React frontend and REST APIs for government compliance.
        He developed a Judge Gymnasts App using React Native and Expo for cross-platform judging.
        Luis built an Enterprise Digital Billing API with Java Spring Boot and AWS integration.
        He created a Multi-Branch ERP System using C# .NET with cloud SQL database.
        Luis is working on an AI Voice Translation System for anime dubbing using sequence-to-sequence models.
        He contributed to an Open-Source Healthcare AI project with Omdena for disease prediction in Africa.
        Luis specializes in computer vision, natural language processing, and audio processing.
        He has experience with AWS, Google Cloud Platform, and DigitalOcean for cloud deployments.
        Luis uses Docker, Jenkins, and CI/CD pipelines for deployment automation.
        He knows Python, Java, C++, JavaScript, and C# programming languages.
        Luis is learning R for statistical analysis and data science.
        His contact email is alexmtzai2002@gmail.com for professional inquiries.
        Luis completed certifications in AI/ML from Zero to Mastery Academy.
        He earned certifications in PyTorch and TensorFlow for deep learning.
        Luis completed algorithms and data structures course from Zero to Mastery.
        He finished machine learning basics and data science courses from Omdena Academy.
        Luis completed AWS for developers and Linux courses from LinkedIn Learning.
        He worked on leveraging NLP in medical prescription project with Omdena Local Chapter.
        Luis has experience with sentence transformers and FAISS for vector databases.
        He uses React for frontend development and Django for Python web applications.
        Luis deploys applications on his own servers with Nginx and Apache configurations.
        He has experience with reverse proxy setups and SSL certificate management.
        Luis founded My Software SV and serves as CEO managing software development projects.
        He has worked with 20+ clients across multiple industries with 97% satisfaction rate.
        Luis achieved 90% on-time project delivery rate and reduced system errors by 25%.
        He specializes in quantized model deployment for CPU-only execution environments.
        Luis has experience with early adopter technologies and cutting-edge AI implementations.
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
            'skills': "Luis specializes in AI/ML with TensorFlow, PyTorch, Python, Java, C++, JavaScript, and cloud technologies like AWS and DigitalOcean.",
            'experience': "Luis has 3+ years of experience, completed 15+ projects, and worked with 20+ clients with 97% satisfaction rate.",
            'education': "Luis studied Software Engineering at Universidad Centroamericana (UCA) for 5 years and has multiple AI/ML certifications.",
            'company': "Luis founded and runs My Software SV, his own software company based in El Salvador.",
            'projects': "Luis has built SaaS billing systems, ERP systems, mobile apps, AI translation systems, and healthcare AI projects.",
            'contact': "You can reach Luis at alexmtzai2002@gmail.com for professional inquiries.",
            'interests': "Luis enjoys boxing, tennis, cooking, and exploring new technologies as an early adopter.",
            'location': "Luis is based in El Salvador and operates his company My Software SV from there.",
            'certifications': "Luis has certifications in AI/ML, PyTorch, TensorFlow, algorithms, and AWS from Zero to Mastery and other academies.",
            'technologies': "Luis works with Python, TensorFlow, PyTorch, React, Java Spring Boot, C# .NET, Docker, and cloud platforms."
        }
    
    def find_best_response(self, query):
        """Find best response using simple matching + embeddings"""
        
        query_lower = query.lower()
        
        # Enhanced keyword matching (very fast)
        if any(word in query_lower for word in ['name', 'who is', 'called', 'who are you']):
            return self.responses['name']
        elif any(word in query_lower for word in ['age', 'old', 'years old']):
            return self.responses['age']
        elif any(word in query_lower for word in ['job', 'work', 'profession', 'engineer', 'what do you do']):
            return self.responses['profession']
        elif any(word in query_lower for word in ['skill', 'technology', 'programming', 'languages', 'tech stack']):
            return self.responses['skills']
        elif any(word in query_lower for word in ['experience', 'worked', 'professional']):
            return self.responses['experience']
        elif any(word in query_lower for word in ['education', 'study', 'university', 'uca', 'degree']):
            return self.responses['education']
        elif any(word in query_lower for word in ['company', 'business', 'software sv', 'founder']):
            return self.responses['company']
        elif any(word in query_lower for word in ['project', 'built', 'created', 'developed', 'portfolio']):
            return self.responses['projects']
        elif any(word in query_lower for word in ['contact', 'email', 'reach', 'hire']):
            return self.responses['contact']
        elif any(word in query_lower for word in ['hobby', 'interest', 'enjoy', 'boxing', 'tennis', 'cooking']):
            return self.responses['interests']
        elif any(word in query_lower for word in ['location', 'where', 'salvador', 'based']):
            return self.responses['location']
        elif any(word in query_lower for word in ['certification', 'certificate', 'course', 'training']):
            return self.responses['certifications']
        elif any(word in query_lower for word in ['tensorflow', 'pytorch', 'react', 'python', 'java']):
            return self.responses['technologies']
        
        # If no keyword match, use semantic search (slower but more accurate)
        try:
            query_embedding = self.embedding_model.encode([query])
            similarities = np.dot(query_embedding, self.embeddings.T)[0]
            best_idx = np.argmax(similarities)
            
            if similarities[best_idx] > 0.3:  # Threshold for relevance
                return f"{self.chunks[best_idx]}"
            else:
                return "I don't have specific information about that. You can contact Luis directly at alexmtzai2002@gmail.com for more details."
                
        except Exception as e:
            return "I'm having trouble processing that question right now."

# Initialize the ultra-light RAG system
rag_system = UltraLightRAG()

# Flask app with CORS
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"📝 Received query: {query}")
        
        # Get response from ultra-light system
        answer = rag_system.find_best_response(query)
        
        print(f"✅ Generated answer: {answer[:100]}...")
        
        response_data = {
            "answer": answer,
            "query": query,
            "model": "Ultra-Lightweight RAG (416MB RAM optimized)",
            "method": "keyword_matching + embeddings",
            "server": "DigitalOcean RAG Server"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/ask_short', methods=['POST', 'OPTIONS'])
def ask_short():
    """Ultra-short responses"""
    if request.method == 'OPTIONS':
        return '', 200
    
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
        "optimization": "keyword_matching + sentence_transformers",
        "cors": "enabled",
        "server": "DigitalOcean"
    })

@app.route('/memory', methods=['GET'])
def memory_info():
    """Check memory usage"""
    try:
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
    except Exception as e:
        return jsonify({
            "error": f"Could not get memory info: {e}",
            "estimated_usage": "~150MB"
        })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        "message": "RAG server is working!",
        "status": "success",
        "endpoints": ["/ask", "/ask_short", "/health", "/memory", "/test"]
    })

# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚨 Starting Ultra-Lightweight RAG for 416MB RAM")
    print("📊 Expected memory usage: ~150MB")
    print("🌐 CORS enabled for all origins")
    
    # Load the system
    if rag_system.load_system():
        print("✅ System loaded successfully!")
        print("🌐 Server starting on http://0.0.0.0:5000")
        print("\n📋 Endpoints:")
        print("  POST /ask - Ask questions")
        print("  POST /ask_short - Short responses") 
        print("  GET /health - System health")
        print("  GET /memory - Memory usage")
        print("  GET /test - Test connection")
        print("\n🔗 CORS Headers:")
        print("  Access-Control-Allow-Origin: *")
        print("  Access-Control-Allow-Methods: GET, POST, OPTIONS")
        print("  Access-Control-Allow-Headers: Content-Type, Authorization")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load system")