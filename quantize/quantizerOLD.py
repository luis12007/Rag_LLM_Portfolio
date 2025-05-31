from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import json

def download_and_quantize():
    # Superior models - ordered from smaller to larger
    model_candidates = [
        # Smaller, efficient models first
        "microsoft/Phi-3.5-mini-instruct",     # 3.8B parameters - efficient
        "google/gemma-2-2b-it",                # 2B parameters - good for testing
        
        # Medium models
        "Qwen/Qwen2.5-7B-Instruct",           # 7B parameters - very good
        "mistralai/Mistral-7B-Instruct-v0.3", # 7B parameters - very efficient
        "google/gemma-2-9b-it",               # 9B parameters - very good performance
        
        # Larger models (comment out if memory issues)
        # "Qwen/Qwen2.5-14B-Instruct",          # 14B parameters - excellent
        # "meta-llama/Llama-3.1-8B-Instruct",   # 8B parameters - great performance
        
        # Code-specific models
        "Qwen/Qwen2.5-Coder-7B-Instruct",     # Excellent for coding tasks
    ]
    
    for model_name in model_candidates:
        try:
            print(f"Trying to download {model_name}...")
            
            # Try with quantization first, fallback to float16 if it fails
            try:
                # Create quantization config for better memory efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,                # 4-bit quantization for maximum compression
                    bnb_4bit_quant_type="nf4",       # NormalFloat4 quantization
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,   # Double quantization for more memory savings
                )
                
                # Download tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Download model with advanced quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="cpu",  # Force CPU for compatibility
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                quantization_type = "4-bit NF4"
                print("✅ Applied 4-bit quantization")
                
            except Exception as quant_error:
                print(f"⚠️ Quantization failed: {quant_error}")
                print("Falling back to float16...")
                
                # Fallback to float16 without quantization
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                quantization_type = "float16"
                print("✅ Applied float16 optimization")
            
            # Save locally
            model_safe_name = model_name.replace("/", "_").replace("-", "_")
            output_dir = f"./quantized_{model_safe_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
            
            # Save model info
            model_info = {
                "original_name": model_name,
                "quantization": quantization_type,
                "size_reduction": "~75%" if "4-bit" in quantization_type else "~50%",
                "recommended_use": "CPU inference with optimized performance",
                "parameters": "Check model card for exact parameter count"
            }
            
            with open(f"{output_dir}/model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            print(f"✅ Successfully processed and saved to: {output_dir}")
            print(f"📊 Model: {model_name}")
            print(f"💾 Location: {output_dir}")
            print(f"🔧 Optimization: {quantization_type}")
            
            return output_dir
            
        except Exception as e:
            print(f"❌ Failed with {model_name}: {e}")
            continue
    
    print("All model downloads failed")
    return None

def download_specific_model(model_name, quantization_level="auto"):
    """Download a specific model with chosen quantization level"""
    
    print(f"Downloading {model_name} with {quantization_level} optimization...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if quantization_level == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            quant_type = "4bit"
            
        elif quantization_level == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            quant_type = "8bit"
            
        else:  # float16 or auto
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            quant_type = "float16"
        
        # Save locally
        model_safe_name = model_name.replace("/", "_").replace("-", "_")
        output_dir = f"./optimized_{quant_type}_{model_safe_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
        # Save model info
        model_info = {
            "original_name": model_name,
            "optimization": quant_type,
            "size_reduction": {"4bit": "~75%", "8bit": "~50%", "float16": "~50%"}[quant_type],
            "recommended_use": "CPU inference"
        }
        
        with open(f"{output_dir}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ {model_name} saved to: {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

if __name__ == "__main__":
    print("Choose download option:")
    print("1. Auto-download best available model")
    print("2. Download specific model")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("\nRecommended models:")
        print("1. microsoft/Phi-3.5-mini-instruct (3.8B - very efficient)")
        print("2. Qwen/Qwen2.5-7B-Instruct (7B - excellent performance)")
        print("3. google/gemma-2-9b-it (9B - very good)")
        print("4. mistralai/Mistral-7B-Instruct-v0.3 (7B - efficient)")
        print("5. Custom model name")
        
        model_choice = input("Enter model choice (1-5): ").strip()
        
        models = {
            "1": "microsoft/Phi-3.5-mini-instruct",
            "2": "Qwen/Qwen2.5-7B-Instruct", 
            "3": "google/gemma-2-9b-it",
            "4": "mistralai/Mistral-7B-Instruct-v0.3"
        }
        
        if model_choice in models:
            model_name = models[model_choice]
        elif model_choice == "5":
            model_name = input("Enter model name: ").strip()
        else:
            print("Invalid choice")
            exit()
        
        quant_level = input("Optimization level (4bit/8bit/float16/auto): ").strip() or "auto"
        model_path = download_specific_model(model_name, quant_level)
        
    else:
        model_path = download_and_quantize()
    
    if model_path:
        print(f"\n🎉 Model ready for use!")
        print(f"📁 Path: {model_path}")
        print(f"🚀 You can now use this in your RAG system")