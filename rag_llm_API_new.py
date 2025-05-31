from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import gc
import psutil
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# === EFFICIENT LUIS PORTFOLIO RAG+LLM SYSTEM ===

class LuisPortfolioRAG:
    def __init__(self):
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_pipeline = None
        self.chunks = []
        self.embeddings = None
        self.chunk_metadata = []
        self.memory_limit_mb = 1250  # 1.25GB in MB
        self.model_type = "auto"
        
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
"""
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    def is_luis_related_query(self, query):
        """Check if query is about Luis (including all name variations and portfolio)"""
        # All variations of Luis Alexander's name
        name_variations = [
            'luis', 'alexander', 'alex', 'hernández', 'hernandez', 'martinez', 'martínez',
            'luis alexander', 'alexander hernández', 'alex hernández', 'alexander martinez',
            'luis hernández', 'luis martinez', 'alex martinez',
            'software sv', 'my software', 'founder', 'ceo', 'salvador'
        ]
        
        # Portfolio-related keywords (same as asking about Luis)
        portfolio_keywords = [
            'portfolio', 'resume', 'cv', 'profile', 'background', 'about',
            'experience', 'skills', 'projects', 'education', 'contact',
            'work', 'company', 'developer', 'programming', 'technology', 
            'achievements', 'certifications', 'ai engineer', 'software engineer'
        ]
        
        query_lower = query.lower()
        
        # Check for name variations
        name_match = any(name in query_lower for name in name_variations)
        
        # Check for portfolio requests
        portfolio_match = any(keyword in query_lower for keyword in portfolio_keywords)
        
        # Accept if any condition is met or if it's a short general query
        return name_match or portfolio_match or len(query.split()) < 5
    
    def normalize_query_for_search(self, query):
        """Normalize query to handle name variations and portfolio requests"""
        query_lower = query.lower()
        
        # Replace name variations with full name for better search
        name_replacements = {
            'alex ': 'luis alexander ',
            'alexander ': 'luis alexander ',
            ' alex': ' luis alexander',
            ' alexander': ' luis alexander'
        }
        
        normalized = query_lower
        for old, new in name_replacements.items():
            normalized = normalized.replace(old, new)
        
        # Handle portfolio requests - convert to person-focused queries
        portfolio_replacements = {
            'portfolio': 'luis alexander background experience skills projects',
            'resume': 'luis alexander background experience skills',
            'cv': 'luis alexander background experience education',
            'profile': 'luis alexander background information',
            'about': 'luis alexander information background'
        }
        
        for old, new in portfolio_replacements.items():
            if old in normalized:
                normalized = normalized.replace(old, new)
        
        return normalized
    
    def load_system(self):
        """Load the memory-optimized Luis RAG+LLM system for 1.479GB memory"""
        try:
            print("🚀 Loading Memory-Optimized Luis Portfolio RAG+LLM System")
            print("💾 Memory Target: 1.479GB (1479MB)")
            print(f"💾 Initial memory: {self.get_memory_usage():.1f}MB")
            
            # MEMORY OPTIMIZATION 1: Use smaller embedding model
            print("\n📊 Loading compact embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Only ~90MB vs 400MB
            print(f"✅ Embeddings loaded | Memory: {self.get_memory_usage():.1f}MB")
            
            # MEMORY OPTIMIZATION 2: Create optimized Luis embeddings
            print("\n📁 Creating optimized Luis portfolio embeddings...")
            self._create_optimized_luis_embeddings()
            print(f"✅ Luis embeddings created | Memory: {self.get_memory_usage():.1f}MB")
            
            # MEMORY OPTIMIZATION 3: Force garbage collection
            print("\n🧹 Optimizing memory before LLM load...")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load GPT-2 Medium with aggressive memory optimization
            print("\n🧠 Loading GPT-2 Medium with memory optimization...")
            self._load_optimized_llm()
            
            final_memory = self.get_memory_usage()
            memory_percentage = (final_memory / self.memory_limit_mb) * 100
            print(f"\n✅ Luis Portfolio System Ready!")
            print(f"💾 Final Memory: {final_memory:.1f}MB / {self.memory_limit_mb}MB ({memory_percentage:.1f}%)")
            
            if memory_percentage > 95:
                print("⚠️ High memory usage - monitor performance")
            elif memory_percentage > 85:
                print("📊 Good memory usage - optimal performance expected")
            else:
                print("🎯 Excellent memory efficiency")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_optimized_luis_embeddings(self):
        """Create memory-optimized embeddings for Luis's portfolio"""
        
        # OPTIMIZATION: Shorter, more focused chunks to reduce memory
        luis_sections = [
            ("personal", "Luis Alexander Hernández Martínez, 23, AI Engineer, El Salvador, Founder CEO My Software SV"),
            ("about", "Luis passionate software engineer specializing Artificial Intelligence, transforming ideas into products"),
            ("interests", "Luis practices boxing tennis, explores technology trends, early adopter, enjoys cooking"),
            ("education", "Luis studied Software Engineering Universidad Centroamericana UCA El Salvador 5 years"),
            ("company", "Luis Founder CEO My Software SV 3+ years, 15+ solutions, 20+ clients, 97% satisfaction"),
            ("internship", "Luis IT Support Fe y Alegría 2021-2022, 100+ employees, 95% resolution rate"),
            ("programming", "Luis uses Python AI/ML, Java enterprise, C++ system, JavaScript web, C# desktop"),
            ("ai_tech", "Luis uses TensorFlow PyTorch Scikit-learn XGBoost Pandas Matplotlib NumPy"),
            ("web_tech", "Luis uses React Django Node.js Express.js for full-stack development"),
            ("cloud", "Luis uses AWS Google Cloud DigitalOcean Linux Docker Apache Nginx"),
            ("projects", "Luis built SaaS billing, RAG assistant, billing API, ERP system, gymnasts app"),
            ("achievements", "Luis 3+ years experience, 15+ projects, 97% satisfaction, 84% AI accuracy"),
            ("contact", "Luis alexmtzai2002@gmail.com (+503)7752-2702 LinkedIn GitHub portfolio El Salvador")
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
        
        # Create embeddings with lower precision to save memory
        print(f"   Creating embeddings for {len(self.chunks)} optimized chunks...")
        self.embeddings = self.embedding_model.encode(
            self.chunks, 
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # OPTIMIZATION: Convert to float32 to save memory (vs float64)
        self.embeddings = self.embeddings.astype(np.float32)
        
        # Save embeddings
        with open('luis_portfolio_embeddings_optimized.pkl', 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'metadata': self.chunk_metadata
            }, f)
        
        print(f"   ✅ Created {len(self.chunks)} memory-optimized embeddings")
    
    def _load_optimized_llm(self):
        """Load best possible model within 1.479GB limit with modern efficient models"""
        try:
            current_memory = self.get_memory_usage()
            available_memory = self.memory_limit_mb - current_memory
            print(f"   Current memory usage: {current_memory:.1f}MB")
            print(f"   Available for LLM: {available_memory:.1f}MB")
            print(f"   Target: Keep total under {self.memory_limit_mb}MB")
            
            # MODERN EFFICIENT MODELS - Better than GPT-2 Medium with less memory
            model_candidates = []
            
            if available_memory >= 600:
                # Microsoft DialoGPT-medium: Conversational, ~600MB, excellent for Q&A
                model_candidates.append(("microsoft/DialoGPT-medium", "DialoGPT-Medium", "Conversational AI optimized for dialogue"))
                
            if available_memory >= 500:
                # GPT-2 Small with better training: ~500MB, good quality
                model_candidates.append(("gpt2", "GPT-2-Small", "Reliable and well-tested"))
                
            if available_memory >= 400:
                # DistilGPT-2: ~400MB, 97% of GPT-2 performance with 50% size
                model_candidates.append(("distilgpt2", "DistilGPT-2", "Distilled GPT-2 with great efficiency"))
                
            if available_memory >= 300:
                # TinyStories models: Very small but surprisingly good for simple tasks
                model_candidates.append(("roneneldan/TinyStories-33M", "TinyStories-33M", "Surprisingly capable tiny model"))
            
            # Try models in order of preference
            for model_name, display_name, description in model_candidates:
                print(f"   🎯 Trying {display_name}: {description}")
                if self._try_load_specific_model(model_name, display_name):
                    return True
                print(f"   ❌ {display_name} failed, trying next option...")
            
            # Ultimate fallback - use a very small model
            print(f"   🔄 Loading minimal fallback model...")
            return self._try_load_minimal_model()
                
        except Exception as e:
            print(f"   ❌ LLM loading failed: {e}")
            return False
    
    def _try_load_specific_model(self, model_name, display_name):
        """Try to load a specific model with optimization"""
        try:
            print(f"      📥 Loading {display_name} tokenizer...")
            
            # Load tokenizer
            if "DialoGPT" in model_name:
                from transformers import AutoTokenizer
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                self.llm_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            print(f"      🔧 Loading {display_name} model with optimization...")
            
            # Try quantization first if available
            try:
                import bitsandbytes
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                
                if "DialoGPT" in model_name:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True,
                        device_map="auto"
                    )
                else:
                    model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True,
                        device_map="auto"
                    )
                
                print(f"      ✅ 8-bit quantization successful for {display_name}!")
                model_type_suffix = "-quantized"
                
            except Exception:
                # Fallback to standard optimization
                print(f"      🔄 Using standard optimization for {display_name}...")
                
                if "DialoGPT" in model_name:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="cpu"
                    )
                else:
                    model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="cpu"
                    )
                
                model_type_suffix = "-optimized"
            
            # Optimize model settings
            model.config.use_cache = False
            model.config.output_attentions = False
            model.config.output_hidden_states = False
            model.eval()
            
            for param in model.parameters():
                param.requires_grad = False
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.llm_tokenizer,
                device=-1,
                return_full_text=False
            )
            
            gc.collect()
            
            self.model_type = f"{display_name.lower().replace('-', '_')}{model_type_suffix}"
            memory_after = self.get_memory_usage()
            
            print(f"      ✅ {display_name} loaded: {memory_after:.1f}MB total")
            
            if memory_after <= self.memory_limit_mb and self._test_optimized_generation():
                print(f"   🎉 SUCCESS: {display_name} operational within {self.memory_limit_mb}MB!")
                return True
            else:
                print(f"      ❌ {display_name} failed memory or generation test")
                self._cleanup_model()
                return False
                
        except Exception as e:
            print(f"      ❌ {display_name} loading failed: {e}")
            self._cleanup_model()
            return False
    
    def _try_load_minimal_model(self):
        """Load the most minimal model as ultimate fallback"""
        try:
            print(f"      📥 Loading minimal DistilGPT-2...")
            
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Load with minimal configuration
            self.llm_pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                tokenizer=self.llm_tokenizer,
                device=-1,
                return_full_text=False
            )
            
            self.llm_pipeline.model.eval()
            gc.collect()
            
            self.model_type = "distilgpt2_minimal"
            memory_after = self.get_memory_usage()
            
            print(f"      ✅ Minimal model loaded: {memory_after:.1f}MB total")
            
            if self._test_optimized_generation():
                print(f"   ✅ Minimal fallback successful!")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"      ❌ Minimal model failed: {e}")
            return False
    
    def _try_load_gpt2_medium_optimized(self):
        """Try to load GPT-2 Medium with 8-bit quantization"""
        try:
            print(f"      📥 Loading GPT-2 Medium tokenizer...")
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            print(f"      🔧 Loading model with 8-bit quantization...")
            
            try:
                # Try 8-bit quantization first (requires bitsandbytes)
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                
                model = GPT2LMHeadModel.from_pretrained(
                    "gpt2-medium",
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                
                print(f"      ✅ 8-bit quantization successful!")
                
            except Exception as quant_error:
                print(f"      ⚠️ 8-bit quantization failed: {quant_error}")
                print(f"      🔄 Falling back to float16 optimization...")
                
                # Fallback to float16 if quantization fails
                model = GPT2LMHeadModel.from_pretrained(
                    "gpt2-medium",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            
            # Disable all unnecessary features
            model.config.use_cache = False
            model.config.output_attentions = False
            model.config.output_hidden_states = False
            model.eval()
            
            # Disable gradients for all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            # Create pipeline manually for better control
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.llm_tokenizer,
                device=-1,
                return_full_text=False
            )
            
            # Force cleanup
            gc.collect()
            
            self.model_type = "gpt2-medium-quantized"
            memory_after = self.get_memory_usage()
            
            print(f"      ✅ GPT-2 Medium loaded: {memory_after:.1f}MB total")
            
            if memory_after <= self.memory_limit_mb and self._test_optimized_generation():
                print(f"   🎉 SUCCESS: Quantized GPT-2 Medium within {self.memory_limit_mb}MB!")
                return True
            else:
                print(f"      ❌ Memory limit exceeded or test failed")
                self._cleanup_model()
                return False
                
        except Exception as e:
            print(f"      ❌ GPT-2 Medium quantization failed: {e}")
            print(f"      💡 Tip: Install bitsandbytes for better quantization:")
            print(f"          pip install bitsandbytes")
            self._cleanup_model()
            return False
    
    def _try_load_gpt2_small(self):
        """Load GPT-2 Small - good balance of quality and memory efficiency"""
        try:
            print(f"      📥 Loading GPT-2 Small...")
            
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer=self.llm_tokenizer,
                device=-1,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16,
                    "use_cache": False
                },
                return_full_text=False
            )
            
            # Optimize model settings
            if hasattr(self.llm_pipeline.model, 'config'):
                self.llm_pipeline.model.config.use_cache = False
                self.llm_pipeline.model.config.output_attentions = False
                self.llm_pipeline.model.config.output_hidden_states = False
            
            self.llm_pipeline.model.eval()
            for param in self.llm_pipeline.model.parameters():
                param.requires_grad = False
            
            gc.collect()
            
            self.model_type = "gpt2-small"
            memory_after = self.get_memory_usage()
            
            print(f"      ✅ GPT-2 Small loaded: {memory_after:.1f}MB total")
            
            if memory_after <= self.memory_limit_mb and self._test_optimized_generation():
                print(f"   🎉 SUCCESS: GPT-2 Small within {self.memory_limit_mb}MB!")
                return True
            else:
                self._cleanup_model()
                return False
                
        except Exception as e:
            print(f"      ❌ GPT-2 Small failed: {e}")
            self._cleanup_model()
            return False
    
    def _try_load_distilgpt2(self):
        """Load DistilGPT-2 - most memory efficient option"""
        try:
            print(f"      📥 Loading DistilGPT-2 (compact and reliable)...")
            
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                tokenizer=self.llm_tokenizer,
                device=-1,
                model_kwargs={"low_cpu_mem_usage": True},
                return_full_text=False
            )
            
            self.llm_pipeline.model.eval()
            for param in self.llm_pipeline.model.parameters():
                param.requires_grad = False
            
            gc.collect()
            
            self.model_type = "distilgpt2"
            memory_after = self.get_memory_usage()
            
            print(f"      ✅ DistilGPT-2 loaded: {memory_after:.1f}MB total")
            
            if self._test_optimized_generation():
                print(f"   🎉 SUCCESS: DistilGPT-2 operational within {self.memory_limit_mb}MB!")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"      ❌ DistilGPT-2 failed: {e}")
            return False
    
    def _cleanup_model(self):
        """Clean up partially loaded models"""
        try:
            self.llm_pipeline = None
            self.llm_tokenizer = None
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except:
            pass
    
    def _load_fallback_model(self):
        """Deprecated - replaced by smart model selection"""
        return self._try_load_distilgpt2()
    
    def _test_optimized_generation(self):
        """Test optimized generation with memory monitoring"""
        try:
            print("   🧪 Testing optimized generation...")
            
            pre_test_memory = self.get_memory_usage()
            
            test_prompt = "Luis is an AI engineer who"
            result = self.llm_pipeline(
                test_prompt,
                max_new_tokens=15,  # Reduced for memory efficiency
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
            
            post_test_memory = self.get_memory_usage()
            
            if isinstance(result, list) and len(result) > 0:
                generated = result[0]['generated_text'][len(test_prompt):].strip()
                print(f"   ✅ Test successful: '{generated[:30]}...'")
                print(f"   📊 Memory during generation: {post_test_memory - pre_test_memory:.1f}MB increase")
                return True
            else:
                print(f"   ❌ Invalid generation result")
                return False
            
        except Exception as e:
            print(f"   ❌ Test generation failed: {e}")
            return False
    
    def search_luis_info(self, query, top_k=3):
        """Search Luis's portfolio information with improved accuracy and name handling"""
        try:
            # Normalize query to handle name variations and portfolio requests
            normalized_query = self.normalize_query_for_search(query)
            
            print(f"   🔍 Searching for: '{query}' -> normalized: '{normalized_query}'")
            
            # Use normalized query for embedding search
            query_embedding = self.embedding_model.encode([normalized_query])
            similarities = np.dot(query_embedding, self.embeddings.T)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.2:
                    results.append({
                        "content": self.chunks[idx],
                        "similarity": float(similarities[idx]),
                        "category": self.chunk_metadata[idx]["category"]
                    })
            
            # If we don't have enough results, expand search with original query
            if len(results) < 2:
                print(f"   📊 Expanding search with original query...")
                original_embedding = self.embedding_model.encode([query])
                original_similarities = np.dot(original_embedding, self.embeddings.T)[0]
                original_top_indices = np.argsort(original_similarities)[-top_k:][::-1]
                
                for idx in original_top_indices:
                    if original_similarities[idx] > 0.15 and len(results) < top_k:
                        # Avoid duplicates
                        if not any(r["content"] == self.chunks[idx] for r in results):
                            results.append({
                                "content": self.chunks[idx],
                                "similarity": float(original_similarities[idx]),
                                "category": self.chunk_metadata[idx]["category"]
                            })
            
            # If still no results, get general information
            if len(results) == 0:
                print(f"   📋 No specific matches, using general Luis information...")
                # Get the most general information chunks
                general_categories = ['personal', 'about', 'company']
                for i, metadata in enumerate(self.chunk_metadata):
                    if metadata['category'] in general_categories and len(results) < 3:
                        results.append({
                            "content": self.chunks[i],
                            "similarity": 0.3,
                            "category": metadata["category"]
                        })
            
            print(f"   ✅ Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f"❌ Luis info search error: {e}")
            return []
    
    def generate_luis_response(self, query, max_words=100):
        """Generate memory-efficient LLM responses about Luis"""
        try:
            # Ensure LLM is available
            if not self.llm_pipeline:
                return "The AI system is currently unavailable. Please contact Luis directly at alexmtzai2002@gmail.com"
            
            print(f"📝 Processing query: '{query}'")
            
            # Search Luis's portfolio for relevant context
            luis_results = self.search_luis_info(query, top_k=2)  # Reduced for memory
            
            if not luis_results:
                return "No specific information found in Luis Alexander's portfolio. Contact Luis at alexmtzai2002@gmail.com"
            
            # Combine context from search results (optimized)
            context_parts = []
            for result in luis_results:
                context_parts.append(result['content'])
            
            luis_context = " ".join(context_parts)[:300]  # Limit context length for memory
            
            # MEMORY-OPTIMIZED prompt (shorter)
            prompt = f"About Luis: {luis_context[:200]}\n\nQ: {query}\nA: Luis"
            
            print(f"   🎯 Generating with {self.model_type} (memory-optimized)")
            
            # MEMORY-EFFICIENT generation parameters
            try:
                result = self.llm_pipeline(
                    prompt,
                    max_new_tokens=40,  # Reduced for memory efficiency
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    top_k=30,
                    repetition_penalty=1.1,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    return_full_text=False  # Only return generated part
                )
                
                # Force garbage collection after generation
                gc.collect()
                
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0]['generated_text'].strip()
                    
                    # Clean and optimize the response
                    if generated:
                        # Ensure proper start
                        if not generated.startswith('Luis'):
                            generated = "Luis " + generated
                        
                        # Clean sentence structure
                        generated = self._clean_optimized_response(generated)
                        
                        # Word limit
                        words = generated.split()
                        if len(words) > max_words:
                            generated = ' '.join(words[:max_words-5]) + "."
                        
                        # Final cleanup
                        if not generated.endswith('.'):
                            generated = generated.rstrip(',;:') + "."
                        
                        print(f"✅ Generated response: {len(generated.split())} words")
                        return generated
                
            except Exception as gen_error:
                print(f"⚠️ Generation error: {gen_error}")
                # Force cleanup
                gc.collect()
            
            # Fallback response
            return self._generate_memory_safe_fallback(luis_context, query)
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            gc.collect()  # Cleanup on error
            return "Unable to generate response about Luis Alexander. Contact: alexmtzai2002@gmail.com"
    
    def _clean_optimized_response(self, text):
        """Clean response with memory efficiency in mind"""
        try:
            # Remove artifacts
            text = text.replace("A: Luis", "").strip()
            text = text.replace("Q:", "").strip()
            
            # Ensure proper sentence structure
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]
            
            return text
            
        except:
            return text
    
    def _generate_memory_safe_fallback(self, context, query):
        """Generate safe fallback response with minimal memory usage"""
        try:
            # Extract key facts efficiently
            context_lower = context.lower()
            
            base_info = "Luis Alexander Hernández Martínez is a 23-year-old AI Engineer from El Salvador"
            
            if 'founder' in context_lower and 'ceo' in context_lower:
                base_info += " and Founder & CEO of My Software SV"
            
            if '3+' in context and 'years' in context_lower:
                base_info += " with 3+ years of experience"
            
            base_info += ". Contact Luis at alexmtzai2002@gmail.com for more information."
            
            return base_info
            
        except:
            return "Luis Alexander Hernández Martínez is an AI Engineer from El Salvador. Contact: alexmtzai2002@gmail.com"

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
                "answer": "This system only provides information about Luis Alexander Hernández Martínez. Please ask questions about Luis.",
                "query": query,
                "method": "luis_filter",
                "restricted": True
            })
        
        # Generate memory-optimized response about Luis
        response = luis_system.generate_luis_response(query)
        
        return jsonify({
            "answer": response,
            "query": query,
            "method": "memory_optimized_generation",
            "model": luis_system.model_type,
            "memory_usage_mb": round(luis_system.get_memory_usage(), 1),
            "memory_limit_mb": luis_system.memory_limit_mb,
            "memory_efficiency": f"{(luis_system.get_memory_usage()/luis_system.memory_limit_mb)*100:.1f}%",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Force cleanup on error
        gc.collect()
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
            "search_scope": "luis_portfolio_only",
            "memory_usage_mb": round(luis_system.get_memory_usage(), 1)
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
        "system_focus": "Luis Alexander Hernández Martínez Portfolio Only",
        "memory_usage_mb": round(luis_system.get_memory_usage(), 1),
        "memory_optimization": "Enabled for 1.479GB limit"
    })

@app.route('/system_status', methods=['GET'])
def get_system_status():
    """Get system status with memory optimization details"""
    memory_usage = luis_system.get_memory_usage()
    
    return jsonify({
        "system_name": "Memory-Optimized Luis Portfolio RAG+LLM",
        "focus": "Luis Alexander Hernández Martínez Only",
        "memory_usage_mb": round(memory_usage, 1),
        "memory_limit_mb": luis_system.memory_limit_mb,
        "memory_efficiency": f"{(memory_usage/luis_system.memory_limit_mb)*100:.1f}%",
        "model": luis_system.model_type,
        "embedding_model": "all-MiniLM-L6-v2 (compact)",
        "optimizations": [
            "Compact embedding model (90MB vs 400MB)",
            "Shorter knowledge chunks",
            "Float32 embeddings",
            "GPT-2 Medium with float16",
            "Disabled model caching",
            "Aggressive garbage collection",
            "Reduced generation tokens"
        ],
        "response_limit": "100 words maximum",
        "status": "operational" if memory_usage < luis_system.memory_limit_mb * 0.9 else "high_memory"
    })

@app.route('/memory_stats', methods=['GET'])
def get_memory_stats():
    """Get detailed memory statistics"""
    try:
        current_memory = luis_system.get_memory_usage()
        memory_percentage = (current_memory / luis_system.memory_limit_mb) * 100
        
        # Estimate component memory usage
        embeddings_size = luis_system.embeddings.nbytes / (1024 * 1024) if luis_system.embeddings is not None else 0
        
        return jsonify({
            "current_memory_mb": round(current_memory, 1),
            "memory_limit_mb": luis_system.memory_limit_mb,
            "memory_percentage": round(memory_percentage, 1),
            "memory_available_mb": round(luis_system.memory_limit_mb - current_memory, 1),
            "estimated_components": {
                "embedding_model_mb": "~90",
                "llm_model_mb": "~350-500 (depending on model)",
                "embeddings_data_mb": round(embeddings_size, 1),
                "system_overhead_mb": "~50-100"
            },
            "status": "within_limit" if current_memory < luis_system.memory_limit_mb else "over_limit",
            "optimization_active": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Efficient Luis Portfolio RAG+LLM System")
    print("👤 Focus: Luis Alexander Hernández Martínez ONLY")
    print("💾 Memory Limit: 1.479GB (1479MB)")
    print("🎯 Smart Model Selection (Better than GPT-2 Medium, Less Memory):")
    print("   • First choice: DialoGPT-Medium (~600MB) - Conversational AI")
    print("   • Second choice: GPT-2 Small (~500MB) - Reliable quality")
    print("   • Third choice: DistilGPT-2 (~400MB) - 97% GPT-2 performance")
    print("   • Fallback: TinyStories (~300MB) - Surprisingly capable")
    print("⚡ Model Benefits:")
    print("   • DialoGPT: Trained specifically for dialogue/Q&A")
    print("   • DistilGPT-2: 97% of GPT-2 performance with 50% size")
    print("   • All models: Better efficiency than GPT-2 Medium")
    print("   • 8-bit quantization available for all models")
    
    # Check for bitsandbytes
    try:
        import bitsandbytes
        print("   ✅ BitsAndBytes detected - 8-bit quantization available")
    except ImportError:
        print("   ⚠️ BitsAndBytes not found")
        print("   💡 For even better memory efficiency: pip install bitsandbytes")
    
    if luis_system.load_system():
        current_memory = luis_system.get_memory_usage()
        memory_percentage = (current_memory / luis_system.memory_limit_mb) * 100
        
        print("✅ Luis Portfolio System Ready!")
        print(f"💾 Memory: {current_memory:.1f}MB / {luis_system.memory_limit_mb}MB ({memory_percentage:.1f}%)")
        print(f"📊 Luis Knowledge Chunks: {len(luis_system.chunks)}")
        print(f"🧠 Model: {luis_system.model_type}")
        
        if current_memory <= luis_system.memory_limit_mb:
            print(f"🎉 SUCCESS: Within {luis_system.memory_limit_mb}MB memory limit!")
            
            # Show what model we got with better descriptions
            if "diallogpt" in luis_system.model_type.lower():
                print(f"🏆 EXCELLENT: Got DialoGPT-Medium - optimized for conversations, great quality!")
            elif "quantized" in luis_system.model_type:
                print(f"🏆 EXCELLENT: Got quantized model - great quality with minimal memory!")
            elif "gpt2_small" in luis_system.model_type:
                print(f"👍 GOOD: Got GPT-2 Small - reliable and efficient!")
            elif "distilgpt2" in luis_system.model_type:
                print(f"✅ SOLID: Got DistilGPT-2 - efficient and capable!")
            elif "tinystories" in luis_system.model_type:
                print(f"⚡ COMPACT: Got TinyStories - surprisingly capable for its size!")
            else:
                print(f"✅ SUCCESS: Got {luis_system.model_type} - operational and efficient!")
        else:
            print(f"⚠️ WARNING: Slightly over memory limit by {current_memory - luis_system.memory_limit_mb:.1f}MB")
        
        print(f"\n🌐 Server running on http://localhost:5000")
        print(f"\n📋 Endpoints:")
        print(f"   POST /ask - Ask questions about Luis (quantized responses)")
        print(f"   POST /search_luis - Search Luis's portfolio")
        print(f"   GET /luis_info - Get Luis summary")
        print(f"   GET /system_status - System status with quantization details")
        print(f"   GET /memory_stats - Detailed memory statistics")
        
        print("\n🧪 Test Commands:")
        print('   curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"query": "Who is Luis?"}\'')
        print('   curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"query": "Tell me about Alexander"}\'')
        print('   curl http://localhost:5000/memory_stats')
        
        print(f"\n🎯 System Features:")
        print(f"   ✅ 8-bit quantization for GPT-2 Medium (if bitsandbytes available)")
        print(f"   ✅ Progressive model loading (Medium → Small → DistilGPT-2)")
        print(f"   ✅ Memory-optimized generation")
        print(f"   ✅ Real-time memory monitoring")
        print(f"   ✅ Automatic cleanup on failures")
        print(f"   ✅ Fallback to unquantized models if needed")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load quantized Luis portfolio system")
        print("💡 Try installing bitsandbytes: pip install bitsandbytes")