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
        
        # Your complete portfolio data
        portfolio_data = """
        Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador.
        He is the Founder and CEO of My Software SV, his own software company.
        Luis specializes in Artificial Intelligence and machine learning with a deep passion for transforming innovative ideas into functional products.
        He considers himself an AI Engineer first and foremost, though his expertise spans the full spectrum of software development.
        Luis is an early adopter of new technologies and always eager to explore cutting-edge solutions.
        He enjoys boxing for physical fitness and discipline, tennis for strategic thinking and coordination.
        Luis loves exploring new technology trends and cooking as creative expression and experimentation.
        He studied Software Engineering at Universidad Centroamericana (UCA) in El Salvador for 5 years.
        Luis built a strong foundation in computer science and developed expertise across multiple domains of software development.
        
        As Founder and CEO of My Software SV from 2022 to present, Luis has developed 15+ comprehensive software solutions.
        He has worked with 20+ clients across multiple industries building SaaS products, ERP systems, CRM solutions, desktop applications, REST APIs, and mobile applications.
        Luis achieved 97% client satisfaction rate with 90% on-time project delivery.
        He reduced system errors by 25% through industry best practices implementation.
        Luis worked as IT Support Specialist at Fe y Alegría El Salvador from August 2021 to January 2022 as an internship.
        He delivered comprehensive technical support to over 100 employees and achieved 95% first-call resolution rate.
        Luis streamlined system access protocols and optimized resource allocation, boosting departmental productivity by 15%.
        
        Luis's programming languages include Python as primary language for AI/ML development.
        He uses Java for enterprise applications and Spring Boot APIs.
        Luis knows C++ for system programming and performance-critical applications.
        He uses JavaScript for full-stack web development and C# for desktop applications and .NET development.
        Luis is currently learning R for statistical analysis.
        
        His AI and Machine Learning technologies include TensorFlow for deep learning model development.
        Luis uses PyTorch for neural network research and implementation.
        He works with Scikit-learn for traditional machine learning algorithms and XGBoost for gradient boosting.
        Luis uses Pandas for data manipulation and analysis, Matplotlib for data visualization, and NumPy for numerical computing.
        
        For web development, Luis uses React for frontend development and SPA creation.
        He works with Django as Python web framework for robust applications.
        Luis uses Node.js for backend JavaScript development and Express.js for RESTful API development.
        
        His cloud and infrastructure experience includes AWS Amazon Web Services for cloud deployment.
        Luis uses Google Cloud Platform for cloud services and AI tools.
        He manages DigitalOcean for VPS hosting and server management.
        Luis has extensive Linux server administration and deployment experience.
        He uses Docker for containerization and deployment, Apache for web server configuration, and Nginx for reverse proxy and load balancing.
        
        Luis's DevOps and automation skills include Jenkins for CI/CD pipeline automation.
        He uses Git for version control and collaboration, implements automated testing for quality assurance.
        Luis specializes in server configuration including Nginx, Apache, and proxy setup.
        
        For mobile development, Luis uses React Native for cross-platform mobile development.
        He works with Expo as React Native development platform.
        Luis has Android native development knowledge and iOS cross-platform development experience.
        
        Luis built a SaaS Billing System using React, REST API, and custom server deployment.
        This full-stack billing system includes government compliance standards with automated invoice generation and management.
        The system features real-time payment tracking and notifications, multi-API integration, government compliance standards, and secure user authentication.
        
        He created a RAG Portfolio Assistant using LangChain, quantized LLM, and DigitalOcean deployment.
        This intelligent AI assistant is trained on portfolio data with self-quantized LLM optimized for CPU execution.
        The system runs 24/7 on DigitalOcean infrastructure with smart filtering for portfolio-related questions only.
        It includes vector similarity search for accurate information retrieval and context-aware response generation.
        
        Luis built an Enterprise Digital Billing API using Java Spring Boot, AWS, and Nginx.
        This enterprise-grade API for digital billing integration features custom authentication token system.
        The system includes multi-server architecture with AWS cloud integration, SMTP server integration with port restriction workarounds.
        It has direct government API connectivity for compliance and comprehensive logging and monitoring systems.
        
        He created a Multi-Branch ERP System using C# .NET, Visual Studio Code, and Cloud SQL Database.
        This comprehensive ERP solution for small businesses includes cashier POS system for efficient customer service.
        The system features multi-branch inventory management for 3 branches, HR system for user administration and payroll.
        It includes ingredients and recipe management and business intelligence and analytics modules.
        
        Luis developed a Judge Gymnasts App using React Native and Expo.
        This cross-platform application for gymnastics judges includes custom authentication system and file management.
        The app features MAG/WAG calculators for scoring, digital whiteboard for notes and diagrams.
        It includes PDF export functionality and responsive interface for tablets and mobile devices.
        
        He is working on an AI Voice Translation System using sequence-to-sequence models, voice recognition, and TTS.
        This advanced AI system for Japanese to English anime dubbing has voice recognition pipeline 100% complete.
        The speech-to-text conversion framework is 95% complete, AI sequence-to-sequence translation model is 85% complete.
        Text-to-speech synthesis engine is 80% complete and voice cloning and replication system is 70% complete.
        
        Luis contributed to Open-Source Healthcare AI project with Omdena using machine learning, Python, and Streamlit.
        This disease prediction model for African healthcare included data collection from multiple African health databases.
        He performed comprehensive exploratory data analysis, feature engineering and data preprocessing.
        Luis worked on model development using ensemble methods, results visualization and interpretation, achieving 84% accuracy in disease prediction.
        
        Luis specializes in supervised learning for classification and regression problems.
        He has experience with unsupervised learning for clustering and dimensionality reduction.
        Luis has basic understanding and implementation of reinforcement learning and deep learning with neural networks and advanced architectures.
        
        His specialized AI areas include computer vision for image processing and recognition.
        Luis works with natural language processing for text analysis and understanding.
        He has experience with audio processing for speech-to-text and text-to-speech.
        Luis implements large language models for optimization, time series prediction for forecasting, and voice recognition for speech processing.
        
        His AI implementation experience includes custom model development and fine-tuning.
        Luis specializes in model optimization and quantization for CPU execution.
        He has production deployment experience with real-world AI system implementation.
        Luis works with RAG systems and retrieval-augmented generation architecture, embedding AI into existing systems.
        
        Luis completed Bootcamp Fundamentals in AI/ML from Zero To Mastery Academy in June 2024.
        He finished Complete A.I. & Machine Learning, Data Science Course from Zero to Mastery Academy in January 2025.
        Luis completed Algorithms and Data Structures from Zero to Mastery Academy in December 2024.
        He finished Leveraging NLP in Medical Prescription from OMDENA Local Chapter in May 2025.
        Luis completed AWS for Developers, Linux, SQL and more from LinkedIn Learning.
        He finished PyTorch for Deep Learning Bootcamp from Zero To Mastery in April 2025.
        Luis completed TensorFlow for Deep Learning Bootcamp from Zero To Mastery in June 2025.
        He finished Machine Learning Basics from Omdena Academy in November 2024.
        Luis completed Data Science and Python Basics from Omdena Academy in December 2024.
        
        Luis has 3+ years of professional software development experience.
        He completed 15+ projects successfully and founded 1 company that is successfully operating.
        Luis served 20+ clients across multiple industries and maintained 97% client satisfaction rate.
        He achieved 25% reduction in system errors through best practices and 84% accuracy in healthcare AI project.
        
        Luis can be contacted at alexmtzai2002@gmail.com for professional inquiries.
        His LinkedIn profile is https://www.linkedin.com/in/alexmtzai/ and GitHub is https://github.com/luis12007.
        Luis's phone number is (+503) 7752-2702 and he is based in El Salvador.
        His complete portfolio is available at https://portfolio-production-319e.up.railway.app.
        
        Luis has extensive server administration experience with Linux, Apache and Nginx servers, reverse proxies, and cloud deployments.
        He specializes in API development with Java Spring Boot and Express.js, authentication systems, and third-party service integration.
        Luis has database management experience with SQL and NoSQL databases including MySQL, PostgreSQL, MongoDB, and cloud solutions.
        He focuses on security implementation with secure authentication, data encryption, and government compliance standards.
        Luis excels at problem solving with creative solutions like working around SMTP port restrictions and optimizing AI models for CPU execution.
        He has team collaboration experience with international teams and open-source projects with 50+ data scientists worldwide.
        Luis has business acumen as a company founder understanding both technical and business sides of software development.
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
            'name': "Luis Alexander Hernández Martínez is a 23-year-old AI Engineer and Founder & CEO of My Software SV from El Salvador.",
            'age': "Luis is 23 years old.",
            'profession': "Luis is an AI Engineer and Software Engineer who founded and runs My Software SV. He specializes in transforming innovative ideas into functional AI products.",
            'skills': "Luis specializes in AI/ML with TensorFlow, PyTorch, Python, Java, C++, JavaScript, C#, React, Django, AWS, DigitalOcean, Docker, and Jenkins.",
            'experience': "Luis has 3+ years of experience as Founder & CEO of My Software SV, completed 15+ projects, worked with 20+ clients with 97% satisfaction rate and 90% on-time delivery.",
            'education': "Luis studied Software Engineering at Universidad Centroamericana (UCA) for 5 years, building a strong foundation in computer science and software development.",
            'company': "Luis founded and runs My Software SV, his own software company in El Salvador, serving 20+ clients across multiple industries since 2022.",
            'projects': "Luis built a SaaS Billing System, RAG Portfolio Assistant, Enterprise Digital Billing API, Multi-Branch ERP System, Judge Gymnasts App, AI Voice Translation System, and Healthcare AI project.",
            'contact': "You can reach Luis at alexmtzai2002@gmail.com, phone (+503) 7752-2702, LinkedIn: https://www.linkedin.com/in/alexmtzai/, or GitHub: https://github.com/luis12007",
            'interests': "Luis enjoys boxing for physical fitness and discipline, tennis for strategic thinking, cooking as creative expression, and exploring new technology trends as an early adopter.",
            'location': "Luis is based in El Salvador where he operates My Software SV.",
            'certifications': "Luis completed AI/ML bootcamps, PyTorch and TensorFlow courses from Zero to Mastery Academy, NLP medical prescription course from Omdena, and AWS/Linux courses from LinkedIn Learning.",
            'technologies': "Luis works with Python, TensorFlow, PyTorch, Java Spring Boot, C# .NET, React, Django, AWS, DigitalOcean, Docker, Jenkins, Nginx, Apache, and various AI/ML frameworks.",
            'internship': "Luis worked as IT Support Specialist at Fe y Alegría El Salvador from August 2021 to January 2022, supporting 100+ employees with 95% first-call resolution rate.",
            'achievements': "Luis achieved 97% client satisfaction rate, 90% on-time project delivery, 25% reduction in system errors, and 84% accuracy in healthcare AI project.",
            'portfolio': "Luis's complete portfolio with detailed project information and live demonstrations is available at https://portfolio-production-319e.up.railway.app",
            'phone': "Luis's phone number is (+503) 7752-2702",
            'linkedin': "Luis's LinkedIn profile is https://www.linkedin.com/in/alexmtzai/",
            'github': "Luis's GitHub profile is https://github.com/luis12007"
        }
    
    def find_best_response(self, query):
        """Find best response using simple matching + embeddings"""
        
        query_lower = query.lower()
        
        # Enhanced keyword matching (very fast)
        if any(word in query_lower for word in ['name', 'who is', 'called', 'who are you']):
            return self.responses['name']
        elif any(word in query_lower for word in ['age', 'old', 'years old']):
            return self.responses['age']
        elif any(word in query_lower for word in ['job', 'work', 'profession', 'engineer', 'what do you do', 'career']):
            return self.responses['profession']
        elif any(word in query_lower for word in ['skill', 'technology', 'programming', 'languages', 'tech stack', 'tools']):
            return self.responses['skills']
        elif any(word in query_lower for word in ['experience', 'worked', 'professional', 'background']):
            return self.responses['experience']
        elif any(word in query_lower for word in ['education', 'study', 'university', 'uca', 'degree', 'school']):
            return self.responses['education']
        elif any(word in query_lower for word in ['company', 'business', 'software sv', 'founder', 'ceo']):
            return self.responses['company']
        elif any(word in query_lower for word in ['project', 'built', 'created', 'developed', 'portfolio']):
            return self.responses['projects']
        elif any(word in query_lower for word in ['contact', 'email', 'reach', 'hire', 'phone']):
            return self.responses['contact']
        elif any(word in query_lower for word in ['hobby', 'interest', 'enjoy', 'boxing', 'tennis', 'cooking']):
            return self.responses['interests']
        elif any(word in query_lower for word in ['location', 'where', 'salvador', 'based']):
            return self.responses['location']
        elif any(word in query_lower for word in ['certification', 'certificate', 'course', 'training', 'bootcamp']):
            return self.responses['certifications']
        elif any(word in query_lower for word in ['tensorflow', 'pytorch', 'react', 'python', 'java', 'django']):
            return self.responses['technologies']
        elif any(word in query_lower for word in ['internship', 'intern', 'fe y alegria', 'support']):
            return self.responses['internship']
        elif any(word in query_lower for word in ['achievement', 'success', 'rate', 'client satisfaction']):
            return self.responses['achievements']
        elif any(word in query_lower for word in ['website', 'portfolio site', 'demo']):
            return self.responses['portfolio']
        elif any(word in query_lower for word in ['phone', 'number', 'call']):
            return self.responses['phone']
        elif any(word in query_lower for word in ['linkedin', 'social', 'network']):
            return self.responses['linkedin']
        elif any(word in query_lower for word in ['github', 'code', 'repository']):
            return self.responses['github']
        
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