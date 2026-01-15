# coding=utf-8
"""
Simple Watermark Demo - Basic Text Watermarking
This is a simplified version of our watermarking system for demonstration purposes.
The complete version will be released after publication.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from Watermark_demo import WatermarkLogitsProcessor, WatermarkDetector

def main():
    print("=== Simple Watermark Demo ===")
    print("Loading model and tokenizer...")
    
    # Load model and tokenizer (using a small model for demo)
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # You can change this to any compatible model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda:1")
    
    # Add pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = model.device
    
    # Initialize watermark processor
    print("Initializing watermark processor...")
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=0.15,
        bigram_table_path="wiki_bigram_table.pkl",
        entropy_skip_percentile=0,
        probability_aware_greenlist=True, 
    )
    
    # Initialize watermark detector
    print("Initializing watermark detector...")
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        device=device,
        tokenizer=tokenizer,
        z_threshold=3.0,
        bigram_table_path="wiki_bigram_table.pkl",
        entropy_skip_percentile=0,
        probability_aware_greenlist=True, 
        normalizers=[],
        ignore_repeated_bigrams=False,
    )
    
    # Demo text generation
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    
    # Tokenize input
    tokenized_input = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Generate watermarked text
    print("Generating watermarked text...")
    with torch.no_grad():
        watermarked_output = model.generate(
            **tokenized_input,
            logits_processor=LogitsProcessorList([watermark_processor]),
            min_new_tokens=200,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Generate non-watermarked text for comparison
    print("Generating non-watermarked text...")
    with torch.no_grad():
        normal_output = model.generate(
            **tokenized_input,
            min_new_tokens=200,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs
    watermarked_text = tokenizer.decode(watermarked_output[0], skip_special_tokens=True)
    normal_text = tokenizer.decode(normal_output[0], skip_special_tokens=True)
    
    print("\n=== Results ===")
    print(f"Watermarked text:\n{watermarked_text}\n")
    print(f"Normal text:\n{normal_text}\n")
    
    # Detect watermark in both texts
    print("=== Watermark Detection ===")
    
    # Detect watermark in watermarked text
    watermark_result = watermark_detector.detect(text=watermarked_text)
    print(f"Watermarked text detection:")
    print(f"  - Prediction: {'WATERMARKED' if watermark_result['prediction'] else 'NOT WATERMARKED'}")
    print(f"  - Z-score: {watermark_result['z_score']:.3f}")
    print(f"  - P-value: {watermark_result['p_value']:.6f}")
    if 'confidence' in watermark_result:
        print(f"  - Confidence: {watermark_result['confidence']:.3f}")
    
    # Detect watermark in normal text
    normal_result = watermark_detector.detect(text=normal_text)
    print(f"\nNormal text detection:")
    print(f"  - Prediction: {'WATERMARKED' if normal_result['prediction'] else 'NOT WATERMARKED'}")
    print(f"  - Z-score: {normal_result['z_score']:.3f}")
    print(f"  - P-value: {normal_result['p_value']:.6f}")
    if 'confidence' in normal_result:
        print(f"  - Confidence: {normal_result['confidence']:.3f}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()