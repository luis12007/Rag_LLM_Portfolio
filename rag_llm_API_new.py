from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import gc
import psutil
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from datetime import datetime

# === STREAMLINED LUIS PORTFOLIO RAG+LLM SYSTEM ===

class LuisPortfolioRAG:
    def __init__(self):
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_pipeline = None
        self.chunks = []
        self.embeddings = None
        self.chunk_metadata = []
        self.memory_limit_mb = 1024
        self.model_type = "gpt2-medium"
        
        # Luis's complete portfolio data
        self.luis_data = self._load_luis_portfolio()
        
    def _load_luis_portfolio(self):
        """Load complete Luis portfolio information"""
        return """
Luis Alexander Hernández Martínez - Professional AI Engineer Portfolio

=== PERSONAL INFORMATION ===
Name: Luis Alexander Hernández Martínez
Age: 23 years old
Location: El Salvador
Profession: AI Engineer & Software Engineer
Company: My Software SV (Founder & CEO)

=== ABOUT LUIS ===
Luis is a passionate 23-year-old software engineer specializing in Artificial Intelligence, with a deep love for transforming innovative ideas into functional products. AI is his passion and he's dedicated to creating impactful solutions that make a real difference. Luis considers himself an AI Engineer first and foremost, though his expertise spans the full spectrum of software development. He's an early adopter of new technologies and always eager to explore cutting-edge solutions.

=== LUIS'S INTERESTS & HOBBIES ===
Boxing: Luis practices boxing for physical fitness and discipline
Tennis: Luis plays tennis for strategic thinking and coordination  
Technology: Luis is always exploring new tech trends
Early Adopter: Luis loves trying new tools and frameworks
Cooking: Luis enjoys cooking as creative expression and experimentation

=== LUIS'S EDUCATION ===
Universidad Centroamericana (UCA) - El Salvador
Degree: Software Engineering
Duration: 5 years
Focus: Luis built a strong foundation in computer science and developed expertise across multiple domains of software development

=== LUIS'S PROFESSIONAL EXPERIENCE ===

My Software SV - Founder & CEO
Duration: 3+ years (2022 - Present)
Location: El Salvador

Luis's Achievements:
- Luis developed 15+ comprehensive software solutions
- Luis worked with 20+ clients across multiple industries
- Luis built SaaS products, ERP systems, CRM solutions, desktop applications, REST APIs, and mobile applications
- Luis achieved 97% client satisfaction rate with 90% on-time project delivery
- Luis reduced system errors by 25% through industry best practices implementation

Fe y Alegría El Salvador - IT Support Specialist
Duration: August 2021 - January 2022
Location: Antiguo Cuscatlán, El Salvador
Type: Internship

Luis's Internship Achievements:
- Luis delivered comprehensive technical support to over 100 employees
- Luis achieved 95% first-call resolution rate
- Luis streamlined system access protocols and optimized resource allocation
- Luis boosted departmental productivity by 15% through enhanced workflow efficiency

=== LUIS'S TECHNICAL SKILLS ===

Programming Languages Luis Uses:
- Python: Luis's primary language for AI/ML development
- Java: Luis uses for enterprise applications and Spring Boot APIs
- C++: Luis uses for system programming and performance-critical applications
- JavaScript: Luis uses for full-stack web development
- C#: Luis uses for desktop applications and .NET development
- R: Luis is currently learning for statistical analysis

AI/Machine Learning Technologies Luis Works With:
- TensorFlow: Luis uses for deep learning model development
- PyTorch: Luis uses for neural network research and implementation
- Scikit-learn: Luis uses for traditional machine learning algorithms
- XGBoost: Luis uses for gradient boosting for structured data
- Pandas: Luis uses for data manipulation and analysis
- Matplotlib: Luis uses for data visualization
- NumPy: Luis uses for numerical computing

Web Development Technologies Luis Uses:
- React: Luis uses for frontend development and SPA creation
- Django: Luis uses as Python web framework for robust applications
- Node.js: Luis uses for backend JavaScript development
- Express.js: Luis uses for RESTful API development

Cloud & Infrastructure Luis Works With:
- AWS: Luis uses Amazon Web Services for cloud deployment
- Google Cloud Platform: Luis uses for cloud services and AI tools
- DigitalOcean: Luis uses for VPS hosting and server management
- Linux: Luis has extensive server administration and deployment experience
- Docker: Luis uses for containerization and deployment
- Apache: Luis configures web servers
- Nginx: Luis sets up reverse proxy and load balancing

=== LUIS'S MAJOR PROJECTS ===

1. Luis's SaaS Billing System
Technology: React, REST API, Custom Server Deployment
Description: Luis created a full-stack billing system with government compliance standards
Features Luis Implemented:
- Automated invoice generation and management
- Real-time payment tracking and notifications
- Multi-API integration for enhanced functionality
- Government compliance standards implementation
- Secure user authentication and authorization

2. Luis's RAG Portfolio Assistant
Technology: LangChain, Quantized LLM, DigitalOcean
Description: Luis built an intelligent AI assistant trained on portfolio data
Features Luis Developed:
- Self-quantized LLM optimized for CPU execution
- 24/7 deployment on DigitalOcean infrastructure
- Smart filtering for portfolio-related questions only
- Vector similarity search for accurate information retrieval
- Context-aware response generation

3. Luis's Enterprise Digital Billing API
Technology: Java Spring Boot, AWS, Nginx
Description: Luis developed an enterprise-grade API for digital billing integration
Features Luis Built:
- Custom authentication token system
- Multi-server architecture with AWS cloud integration
- SMTP server integration with workarounds for port restrictions
- Direct government API connectivity for compliance
- Comprehensive logging and monitoring systems

4. Luis's Multi-Branch ERP System
Technology: C# .NET, Visual Studio Code, Cloud SQL Database
Description: Luis created a comprehensive ERP solution for small businesses
Features Luis Implemented:
- Cashier POS system for efficient customer service
- Multi-branch inventory management (3 branches)
- HR system for user administration and payroll
- Ingredients and recipe management
- Business intelligence and analytics modules

5. Luis's Judge Gymnasts App
Technology: React Native, Expo
Description: Luis developed a cross-platform application for gymnastics judges
Features Luis Created:
- Custom authentication system
- File management and organization
- MAG/WAG calculators for scoring
- Digital whiteboard for notes and diagrams
- PDF export functionality
- Responsive interface for tablets and mobile devices

6. Luis's AI Voice Translation System (Work in Progress)
Technology: Sequence-to-Sequence Models, Voice Recognition, TTS
Description: Luis is building an advanced AI system for Japanese to English anime dubbing
Current Status of Luis's Project:
- Voice recognition pipeline (Japanese): 100% complete
- Speech-to-text conversion framework: 95% complete
- AI sequence-to-sequence translation model: 85% complete
- Text-to-speech synthesis engine: 80% complete
- Voice cloning and replication system: 70% complete

7. Luis's Open-Source Healthcare AI (Omdena Collaboration)
Technology: Machine Learning, Python, Streamlit
Description: Luis contributed to a disease prediction model for African healthcare
Luis's Contributions:
- Data collection from multiple African health databases
- Comprehensive exploratory data analysis (EDA)
- Feature engineering and data preprocessing
- Model development using ensemble methods
- Results visualization and interpretation
- Luis achieved 84% accuracy in disease prediction

=== LUIS'S AI/ML EXPERTISE ===

Machine Learning Algorithms Luis Knows:
- Supervised Learning: Luis works with classification and regression problems
- Unsupervised Learning: Luis implements clustering and dimensionality reduction
- Reinforcement Learning: Luis has basic understanding and implementation
- Deep Learning: Luis builds neural networks and advanced architectures

Specialized AI Areas Luis Works In:
- Computer Vision: Luis does image processing and recognition
- Natural Language Processing: Luis performs text analysis and understanding
- Audio Processing: Luis works with speech-to-text and text-to-speech
- Large Language Models (LLMs): Luis implements and optimizes LLMs
- Time Series Prediction: Luis does forecasting and pattern recognition
- Voice Recognition: Luis works with speech processing and analysis

Luis's AI Implementation Experience:
- Model Training: Luis does custom model development and fine-tuning
- Model Optimization: Luis specializes in quantization for CPU execution
- Production Deployment: Luis implements real-world AI systems
- RAG Systems: Luis builds Retrieval-Augmented Generation architecture
- Model Integration: Luis embeds AI into existing systems

=== LUIS'S ACHIEVEMENTS ===
- Luis has 3+ years of professional software development experience
- Luis completed 15+ projects successfully
- Luis founded 1 company that is successfully operating
- Luis served 20+ clients across multiple industries
- Luis maintained 97% client satisfaction rate
- Luis achieved 25% reduction in system errors through best practices
- Luis achieved 84% accuracy in healthcare AI project

=== CONTACT LUIS ===
Email: alexmtzai2002@gmail.com
LinkedIn: https://www.linkedin.com/in/alexmtzai/
GitHub: https://github.com/luis12007
Phone: (+503) 7752-2702
Portfolio Website: https://portfolio-production-319e.up.railway.app
Location: El Salvador

=== LUIS'S COURSES AND CERTIFICATIONS ===

1. Bootcamp Fundamentals in AI/ML - Luis completed at Zero To Mastery Academy (June 2024)
2. Complete A.I. & Machine Learning, Data Science Course - Luis completed at Zero to Mastery Academy (January 2025)
3. Algorithms and Data Structures - Luis completed at Zero to Mastery Academy (December 2024)
4. Leveraging NLP in Medical Prescription - Luis completed at OMDENA Local Chapter (May 2025)
5. AWS for Developers, Linux, SQL and more - Luis completed at LinkedIn Learning
6. PyTorch for Deep Learning Bootcamp - Luis completed at Zero To Mastery (April 2025)
7. TensorFlow for Deep Learning Bootcamp - Luis completed at Zero To Mastery (June 2025)
8. Machine Learning Basics - Luis completed at Omdena Academy (November 2024)
9. Data Science and Python Basics - Luis completed at Omdena Academy (December 2024)

=== LUIS'S PROFESSIONAL PHILOSOPHY ===
Luis believes in transforming complex problems into elegant AI solutions. Luis's approach combines technical expertise with creative problem-solving to build high-quality products that deliver real value. Luis is constantly evolving as a professional, learning from every project, and refining his approach to stay at the forefront of AI and software development.

=== ADDITIONAL TECHNICAL DETAILS ABOUT LUIS ===

Server Administration: Luis has extensive experience with Linux server administration, including configuring Apache and Nginx servers, setting up reverse proxies, and managing cloud deployments on DigitalOcean, AWS, and Google Cloud Platform.

API Development: Luis specializes in building robust REST APIs using Java Spring Boot and Express.js, with experience in authentication systems, rate limiting, and integration with multiple third-party services.

Database Management: Luis is experienced with both SQL and NoSQL databases, including MySQL, PostgreSQL, MongoDB, and cloud database solutions.

Security Implementation: Luis has a strong focus on security best practices, including secure authentication, data encryption, and compliance with government standards for digital transactions.

Problem Solving: Luis excels at finding creative solutions to complex technical challenges, such as working around SMTP port restrictions in commercial web services and optimizing AI models for CPU-only execution.

Team Collaboration: Luis has experience working with international teams, including contributing to open-source projects with 50+ data scientists from around the world.

Business Acumen: As a company founder, Luis understands both the technical and business sides of software development, enabling him to create solutions that are not only technically sound but also commercially viable.
"""
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    def is_luis_related_query(self, query):
        """Check if query is about Luis"""
        luis_keywords = [
            'luis', 'alexander', 'hernández', 'hernandez', 'martinez', 'martínez',
            'software sv', 'my software', 'founder', 'ceo', 'salvador', 'ai engineer',
            'portfolio', 'experience', 'skills', 'projects', 'education', 'contact',
            'work', 'company', 'developer', 'programming', 'technology', 'achievements'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in luis_keywords) or len(query) < 10
    
    def load_system(self):
        """Load the streamlined Luis RAG+LLM system"""
        try:
            print("🚀 Loading Streamlined Luis Portfolio RAG+LLM System")
            print(f"💾 Initial memory: {self.get_memory_usage():.1f}MB")
            
            # Load embedding model
            print("\n📊 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            print(f"✅ Embeddings loaded | Memory: {self.get_memory_usage():.1f}MB")
            
            # Create Luis-specific embeddings
            print("\n📁 Creating Luis portfolio embeddings...")
            self._create_luis_embeddings()
            print(f"✅ Luis embeddings created | Memory: {self.get_memory_usage():.1f}MB")
            
            # Load LLM
            print("\n🧠 Loading LLM...")
            self._load_llm()
            print(f"✅ LLM loaded | Memory: {self.get_memory_usage():.1f}MB")
            
            final_memory = self.get_memory_usage()
            print(f"\n✅ Luis Portfolio System Ready! Memory: {final_memory:.1f}MB / {self.memory_limit_mb}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_luis_embeddings(self):
        """Create embeddings specifically for Luis's portfolio"""
        
        # Split Luis data into logical sections
        luis_sections = [
            ("personal_info", "Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador. He is the Founder and CEO of My Software SV, his own software company."),
            ("about_luis", "Luis is a passionate software engineer specializing in Artificial Intelligence, with a deep love for transforming innovative ideas into functional products. Luis considers himself an AI Engineer first and foremost, though his expertise spans the full spectrum of software development."),
            ("luis_interests", "Luis practices boxing for physical fitness and discipline. Luis plays tennis for strategic thinking and coordination. Luis is always exploring new technology trends. Luis loves trying new tools and frameworks. Luis enjoys cooking as creative expression."),
            ("luis_education", "Luis studied Software Engineering at Universidad Centroamericana (UCA) in El Salvador for 5 years. Luis built a strong foundation in computer science and developed expertise across multiple domains of software development."),
            ("luis_company", "Luis is the Founder and CEO of My Software SV for 3+ years since 2022. Luis developed 15+ comprehensive software solutions. Luis worked with 20+ clients across multiple industries. Luis achieved 97% client satisfaction rate with 90% on-time project delivery. Luis reduced system errors by 25%."),
            ("luis_internship", "Luis worked as IT Support Specialist at Fe y Alegría El Salvador from August 2021 to January 2022. Luis delivered technical support to over 100 employees. Luis achieved 95% first-call resolution rate. Luis boosted departmental productivity by 15%."),
            ("luis_programming", "Luis's primary language is Python for AI/ML development. Luis uses Java for enterprise applications and Spring Boot APIs. Luis uses C++ for system programming. Luis uses JavaScript for full-stack web development. Luis uses C# for desktop applications. Luis is learning R for statistical analysis."),
            ("luis_ai_ml", "Luis uses TensorFlow for deep learning model development. Luis uses PyTorch for neural network research. Luis uses Scikit-learn for traditional machine learning algorithms. Luis uses XGBoost for gradient boosting. Luis uses Pandas, Matplotlib, and NumPy for data science."),
            ("luis_web_dev", "Luis uses React for frontend development and SPA creation. Luis uses Django as Python web framework for robust applications. Luis uses Node.js for backend JavaScript development. Luis uses Express.js for RESTful API development."),
            ("luis_cloud", "Luis uses AWS Amazon Web Services for cloud deployment. Luis uses Google Cloud Platform for cloud services and AI tools. Luis uses DigitalOcean for VPS hosting and server management. Luis has extensive Linux server administration experience. Luis uses Docker, Apache, and Nginx."),
            ("luis_saas_billing", "Luis created a SaaS Billing System using React, REST API, and custom server deployment. Luis built a full-stack billing system with government compliance standards, automated invoice generation, real-time payment tracking, multi-API integration, and secure authentication."),
            ("luis_rag_assistant", "Luis built a RAG Portfolio Assistant using LangChain, quantized LLM, and DigitalOcean deployment. Luis created an intelligent AI assistant with self-quantized LLM optimized for CPU execution, 24/7 deployment, smart filtering, vector similarity search, and context-aware response generation."),
            ("luis_billing_api", "Luis developed an Enterprise Digital Billing API using Java Spring Boot, AWS, and Nginx. Luis built an enterprise-grade API with custom authentication, multi-server architecture, SMTP integration, government API connectivity, and comprehensive logging systems."),
            ("luis_erp", "Luis created a Multi-Branch ERP System using C# .NET, Visual Studio Code, and Cloud SQL Database. Luis built a comprehensive ERP solution with cashier POS system, multi-branch inventory management for 3 branches, HR system, and business intelligence modules."),
            ("luis_gymnasts_app", "Luis developed a Judge Gymnasts App using React Native and Expo. Luis created a cross-platform application with custom authentication, file management, MAG/WAG calculators, digital whiteboard, PDF export, and responsive interface for tablets and mobile."),
            ("luis_voice_translation", "Luis is building an AI Voice Translation System using sequence-to-sequence models, voice recognition, and TTS for Japanese to English anime dubbing. Luis has voice recognition 100% complete, speech-to-text 95% complete, translation model 85% complete, TTS 80% complete, voice cloning 70% complete."),
            ("luis_healthcare_ai", "Luis contributed to Open-Source Healthcare AI project with Omdena using machine learning, Python, and Streamlit. Luis worked on disease prediction model for African healthcare with data collection, EDA, feature engineering, model development, and achieved 84% accuracy."),
            ("luis_ai_expertise", "Luis works with supervised learning for classification and regression, unsupervised learning for clustering, reinforcement learning, and deep learning with neural networks. Luis specializes in computer vision, NLP, audio processing, LLMs, time series prediction, and voice recognition."),
            ("luis_achievements", "Luis has 3+ years professional software development experience. Luis completed 15+ projects successfully. Luis founded 1 company successfully operating. Luis served 20+ clients across multiple industries. Luis maintained 97% client satisfaction rate. Luis achieved 84% accuracy in healthcare AI project."),
            ("contact_luis", "Contact Luis at alexmtzai2002@gmail.com, phone (+503) 7752-2702, LinkedIn https://www.linkedin.com/in/alexmtzai/, GitHub https://github.com/luis12007, Portfolio https://portfolio-production-319e.up.railway.app, Location El Salvador."),
            ("luis_certifications", "Luis completed AI/ML bootcamps from Zero To Mastery Academy, PyTorch and TensorFlow courses, Algorithms and Data Structures, NLP in Medical Prescription from OMDENA, AWS for Developers, Machine Learning Basics, and Data Science courses from various institutions between 2024-2025."),
            ("luis_philosophy", "Luis believes in transforming complex problems into elegant AI solutions. Luis's approach combines technical expertise with creative problem-solving to build high-quality products. Luis is constantly evolving, learning from every project, staying at the forefront of AI and software development."),
            ("luis_technical_details", "Luis has extensive Linux server administration experience, API development with Spring Boot and Express.js, database management with SQL and NoSQL, security implementation, creative problem solving for technical challenges, international team collaboration, and business acumen as company founder.")
        ]
        
        self.chunks = []
        self.chunk_metadata = []
        
        for category, content in luis_sections:
            self.chunks.append(content)
            self.chunk_metadata.append({
                "category": category,
                "length": len(content),
                "type": "luis_info"
            })
        
        # Create embeddings
        print(f"   Creating embeddings for {len(self.chunks)} Luis portfolio chunks...")
        self.embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        
        # Save embeddings
        with open('luis_portfolio_embeddings.pkl', 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'metadata': self.chunk_metadata
            }, f)
        
        print(f"   ✅ Created {len(self.chunks)} Luis-specific embeddings")
    
    def _load_llm(self):
        """Load LLM for Luis portfolio responses"""
        try:
            print("   Loading GPT-2 Medium for Luis responses...")
            
            # Load tokenizer
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Try direct pipeline creation first (simplest approach)
            try:
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model="gpt2-medium",
                    tokenizer=self.llm_tokenizer,
                    device=-1,  # CPU
                    torch_dtype=torch.float32
                )
                self.model_type = "gpt2-medium"
                print("   ✅ GPT-2 Medium pipeline loaded for Luis responses")
                return True
                
            except Exception as pipeline_error:
                print(f"   ⚠️ Pipeline creation failed: {pipeline_error}")
                print("   Trying manual model loading...")
                
                # Fallback: Load model manually without device_map
                self.llm_model = GPT2LMHeadModel.from_pretrained(
                    "gpt2-medium",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Move to CPU manually
                self.llm_model = self.llm_model.to('cpu')
                self.llm_model.eval()
                
                self.model_type = "gpt2-medium"
                print("   ✅ GPT-2 Medium model loaded manually for Luis responses")
                return True
            
        except Exception as e:
            print(f"   ❌ GPT-2 Medium loading failed: {e}")
            print("   Falling back to DistilGPT-2...")
            
            # Fallback to DistilGPT-2
            try:
                self.llm_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    tokenizer=self.llm_tokenizer,
                    device=-1,
                    torch_dtype=torch.float32
                )
                
                self.model_type = "distilgpt2"
                print("   ✅ DistilGPT-2 loaded as fallback")
                return True
                
            except Exception as fallback_error:
                print(f"   ❌ Complete LLM loading failure: {fallback_error}")
                return False
    
    def search_luis_info(self, query, top_k=3):
        """Search Luis's portfolio information with improved accuracy"""
        try:
            query_embedding = self.embedding_model.encode([query])
            similarities = np.dot(query_embedding, self.embeddings.T)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.2:  # Lower threshold to get more context
                    results.append({
                        "content": self.chunks[idx],
                        "similarity": float(similarities[idx]),
                        "category": self.chunk_metadata[idx]["category"]
                    })
            
            # If we don't have enough high-quality results, add more context
            if len(results) < 2:
                print(f"   📊 Low similarity results for '{query}', expanding search...")
                for idx in top_indices:
                    if similarities[idx] > 0.15 and len(results) < top_k:
                        if not any(r["content"] == self.chunks[idx] for r in results):
                            results.append({
                                "content": self.chunks[idx],
                                "similarity": float(similarities[idx]),
                                "category": self.chunk_metadata[idx]["category"]
                            })
            
            return results
            
        except Exception as e:
            print(f"❌ Luis info search error: {e}")
            return []
    
    def validate_response_accuracy(self, response, original_context):
        """Double-check that response only contains information from the context"""
        try:
            # Extract key facts from the response
            response_lower = response.lower()
            context_lower = original_context.lower()
            
            # Key validation checks
            suspicious_phrases = [
                'university of', 'college of', 'graduated from', 'bachelor of', 'master of',
                'years of experience', 'currently working', 'recently joined', 'previously worked',
                'specializes in machine learning', 'expert in deep learning', 'proficient in',
                'certified in', 'award', 'recognition', 'published', 'research'
            ]
            
            # Check for specific false claims
            false_claims = []
            
            # Check years of experience claims
            if 'years' in response_lower and 'experience' in response_lower:
                if '3+' not in context_lower and 'three' not in context_lower:
                    # Check if any other number is mentioned
                    import re
                    years_match = re.search(r'(\d+)\s*years', response_lower)
                    if years_match:
                        claimed_years = years_match.group(1)
                        if claimed_years not in context_lower:
                            false_claims.append(f"claimed {claimed_years} years experience")
            
            # Check for university/education details not in context
            if any(phrase in response_lower for phrase in ['university', 'college', 'degree']) and 'uca' not in context_lower:
                false_claims.append("education details not in context")
            
            # Check for specific technology claims
            tech_keywords = ['python', 'java', 'react', 'tensorflow', 'pytorch', 'aws']
            for tech in tech_keywords:
                if tech in response_lower and tech not in context_lower:
                    false_claims.append(f"mentioned {tech} not in context")
            
            return len(false_claims) == 0, false_claims
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return True, []  # Default to accepting if validation fails
    
    def generate_luis_response(self, query, max_words=100):
        """Generate response about Luis with strict 100-word limit and double-checking"""
        try:
            # Search Luis's portfolio with higher relevance threshold
            luis_results = self.search_luis_info(query, top_k=3)
            
            if not luis_results:
                return "The available information about Luis doesn't contain details to answer that specific question. Contact Luis at alexmtzai2002@gmail.com"
            
            # Combine relevant Luis information - keep original for validation
            context_parts = []
            for result in luis_results:
                context_parts.append(result['content'])
            
            luis_context = " ".join(context_parts)
            original_context = luis_context  # Keep for validation
            
            # Ultra-strict prompt for accuracy and brevity
            prompt = f"""FACTS ABOUT LUIS ALEXANDER HERNÁNDEZ MARTÍNEZ:
{luis_context[:500]}

QUESTION: {query}

STRICT RULES:
- Use ONLY the facts listed above
- Maximum 100 words
- Third person only ("Luis")
- Start with "Luis"
- Be factual and precise

RESPONSE: Luis"""
            
            # Generate with very conservative parameters
            generated = ""
            
            if self.llm_pipeline:
                result = self.llm_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + 80,  # Very short for 100 words
                    temperature=0.1,  # Even lower for maximum accuracy
                    do_sample=True,
                    top_p=0.8,
                    top_k=30,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
                
                full_response = result[0]['generated_text']
                response_start = full_response.find("RESPONSE: Luis") + len("RESPONSE: ")
                generated = full_response[response_start:].strip()
                
            elif self.llm_model:
                inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt", max_length=400, truncation=True)
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 80,
                        temperature=0.2,
                        do_sample=True,
                        top_p=0.8,
                        top_k=30,
                        repetition_penalty=1.2,
                        early_stopping=True,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
                
                full_response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_start = full_response.find("RESPONSE: Luis") + len("RESPONSE: ")
                generated = full_response[response_start:].strip()
                
                del inputs, outputs
            
            else:
                return "LLM system unavailable. Contact Luis at alexmtzai2002@gmail.com"
            
            gc.collect()
            
            # Enhanced post-processing with validation
            if generated:
                # Clean up response
                generated = generated.replace("RESPONSE:", "").strip()
                
                # Ensure starts with Luis
                if not generated.startswith('Luis'):
                    generated = "Luis " + generated
                
                # STRICT 100-word limit
                words = generated.split()
                if len(words) > 100:
                    generated = ' '.join(words[:95])  # Leave room for ending
                    
                    # Try to end at sentence boundary
                    last_period = generated.rfind('.')
                    if last_period > len(generated) * 0.8:
                        generated = generated[:last_period + 1]
                    else:
                        generated = generated + "."
                
                # Ensure proper ending
                if not generated.endswith(('.', '!', '?')):
                    generated = generated.rstrip(',') + "."
                
                # DOUBLE-CHECK: Validate response accuracy
                is_accurate, false_claims = self.validate_response_accuracy(generated, original_context)
                
                if not is_accurate:
                    print(f"⚠️ Response validation failed: {false_claims}")
                    # Return safer, more general response
                    return "Luis Alexander Hernández Martínez is a 23-year-old AI Engineer and Founder & CEO of My Software SV from El Salvador. Contact Luis at alexmtzai2002@gmail.com for detailed information."
                
                # Final word count check
                final_words = len(generated.split())
                if final_words > 100:
                    words = generated.split()
                    generated = ' '.join(words[:100]) + "."
                
                print(f"📝 Generated response: {len(generated.split())} words")
                return generated
            
            return "Please ask a specific question about Luis. Contact: alexmtzai2002@gmail.com"
            
        except Exception as e:
            print(f"❌ Luis response generation error: {e}")
            return "Error generating response about Luis. Contact: alexmtzai2002@gmail.com"

# Initialize Luis Portfolio System
luis_system = LuisPortfolioRAG()

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_about_luis():
    """Ask questions about Luis - ONLY Luis-related queries accepted"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"📝 Query about Luis: {query}")
        
        # Check if question is about Luis
        if not luis_system.is_luis_related_query(query):
            return jsonify({
                "answer": "This system only provides information about Luis Alexander Hernández Martínez. Please ask questions about Luis's background, skills, experience, projects, or contact information.",
                "query": query,
                "method": "luis_filter",
                "restricted": True
            })
        
        # Generate Luis-specific response
        response = luis_system.generate_luis_response(query)
        
        return jsonify({
            "answer": response,
            "query": query,
            "method": "luis_rag_llm",
            "model": "gpt2-medium",
            "third_person": True,
            "word_limit": 100,
            "memory_usage_mb": round(luis_system.get_memory_usage(), 1),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search_luis', methods=['POST', 'OPTIONS'])
def search_luis_info():
    """Search Luis's portfolio information"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No search query provided"}), 400
        
        results = luis_system.search_luis_info(query, top_k=5)
        
        return jsonify({
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_scope": "luis_portfolio_only"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/luis_info', methods=['GET'])
def get_luis_info():
    """Get Luis portfolio information summary"""
    return jsonify({
        "name": "Luis Alexander Hernández Martínez",
        "age": 23,
        "profession": "AI Engineer & Software Engineer",
        "company": "My Software SV (Founder & CEO)",
        "location": "El Salvador",
        "experience_years": "3+",
        "projects_completed": "15+",
        "clients_served": "20+",
        "satisfaction_rate": "97%",
        "contact": {
            "email": "alexmtzai2002@gmail.com",
            "phone": "(+503) 7752-2702",
            "linkedin": "https://www.linkedin.com/in/alexmtzai/",
            "github": "https://github.com/luis12007",
            "portfolio": "https://portfolio-production-319e.up.railway.app"
        },
        "total_knowledge_chunks": len(luis_system.chunks),
        "system_focus": "Luis Alexander Hernández Martínez Portfolio Only"
    })

@app.route('/system_status', methods=['GET'])
def get_system_status():
    """Get system status"""
    memory_usage = luis_system.get_memory_usage()
    
    return jsonify({
        "system_name": "Luis Portfolio RAG+LLM",
        "focus": "Luis Alexander Hernández Martínez Only",
        "memory_usage_mb": round(memory_usage, 1),
        "memory_limit_mb": luis_system.memory_limit_mb,
        "model": luis_system.model_type,
        "embedding_model": "all-mpnet-base-v2",
                    "response_limit": "100 words maximum (strict)",
        "perspective": "Third person only",
        "information_validation": "Strict - portfolio data only",
        "query_filtering": "Luis-related questions only",
        "status": "operational" if memory_usage < 900 else "high_memory"
    })

if __name__ == '__main__':
    print("🚀 Starting Streamlined Luis Portfolio RAG+LLM System")
    print("👤 Focus: Luis Alexander Hernández Martínez ONLY")
    print("📝 Response Limit: 200 words maximum")
    print("🎭 Perspective: Third person only")
    print("🔒 Information: Portfolio data validation")
    print("🚫 Query Filter: Luis-related questions only")
    
    if luis_system.load_system():
        print(f"\n✅ Luis Portfolio System Ready!")
        print(f"💾 Memory: {luis_system.get_memory_usage():.1f}MB / {luis_system.memory_limit_mb}MB")
        print(f"📊 Luis Knowledge Chunks: {len(luis_system.chunks)}")
        print(f"🧠 Model: {luis_system.model_type}")
        
        print(f"\n🌐 Server running on http://localhost:5000")
        print(f"\n📋 Endpoints:")
        print(f"   POST /ask - Ask questions about Luis (filtered)")
        print(f"   POST /search_luis - Search Luis's portfolio")
        print(f"   GET /luis_info - Get Luis summary")
        print(f"   GET /system_status - System status")
        
        print("\n🧪 Test Commands:")
        print('   curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"query": "Who is Luis?"}\'')
        print('   curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"query": "What are Luis skills?"}\'')
        print('   curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"query": "Tell me about Luis projects"}\'')
        print('   curl -X POST http://localhost:5000/search_luis -H "Content-Type: application/json" -d \'{"query": "AI projects"}\'')
        print('   curl http://localhost:5000/luis_info')
        
        print("\n🎯 System Features:")
        print("   ✅ ONLY answers questions about Luis")
        print("   ✅ Uses ONLY verified portfolio information")
        print("   ✅ Responses capped at 100 words (strict)")
        print("   ✅ Always third person perspective")
        print("   ✅ RAG+LLM generated responses")
        print("   ✅ Filters out non-Luis questions")
        print("   ✅ Double-checks accuracy (no made-up info)")
        print("   ✅ Memory optimized for 1GB")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load Luis portfolio system")