import pandas as pd
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList)
from Watermark_demo import WatermarkLogitsProcessor, WatermarkDetector
import os
from tqdm import tqdm
import argparse
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import roc_curve, auc
import numpy as np

# Download required NLTK data
nltk.download('punkt')

def setup_model_and_watermark(model_name="meta-llama/Llama-2-7b-chat-hf"):
    """Setup model, tokenizer, and watermark components"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    entropy_skip_percentile_value=0
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=0.2,
        bigram_table_path="wiki_bigram_table.pkl",
        entropy_skip_percentile=entropy_skip_percentile_value,
        probability_aware_greenlist=True,
    )
    
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        device=model.device,
        tokenizer=tokenizer,
        z_threshold=3.0,
        bigram_table_path="wiki_bigram_table.pkl",
        entropy_skip_percentile=entropy_skip_percentile_value,
        probability_aware_greenlist=True,
        normalizers=[],
        ignore_repeated_bigrams=True,
    )
    
    return model, tokenizer, watermark_processor, watermark_detector

def extract_prompt_and_human_text(text, tokenizer, target_human_tokens=200):
    """
    Extract first two sentences as prompt and subsequent 200 tokens as human text
    Args:
        text: Full text from C4
        tokenizer: Tokenizer to count tokens
        target_human_tokens: Target number of tokens for human text (default 200)
    Returns:
        tuple: (prompt, human_text) or (None, None) if extraction fails
    """
    try:
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:  # Need at least 3 sentences (2 for prompt + 1 for human text)
            return None, None
        
        # Use first two sentences as prompt
        prompt = sentences[0] + " " + sentences[1]
        
        # Get remaining text
        remaining_text = " ".join(sentences[2:])
        
        # Tokenize remaining text to get exactly 200 tokens
        remaining_tokens = tokenizer.encode(remaining_text, add_special_tokens=False)
        
        if len(remaining_tokens) < target_human_tokens:
            # If not enough tokens, use all remaining text
            human_text = remaining_text
        else:
            # Take exactly target_human_tokens tokens
            human_tokens = remaining_tokens[:target_human_tokens]
            human_text = tokenizer.decode(human_tokens, skip_special_tokens=True)
        
        # Clean up text
        prompt = prompt.strip()
        human_text = human_text.strip()
        
        if len(prompt) < 10 or len(human_text) < 10:  # Filter out too short texts
            return None, None
            
        return prompt, human_text
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return None, None

def detect_watermark(detector, text):
    """Detect watermark in text and return formatted results"""
    try:
        score_dict = detector.detect(text)
        result = {
            'num_tokens_scored': score_dict.get('num_tokens_scored', 0),
            'num_green_tokens': score_dict.get('num_green_tokens', 0),
            'green_fraction': score_dict.get('green_fraction', 0.0),
            'z_score': score_dict.get('z_score', 0.0),
            'p_value': score_dict.get('p_value', 1.0),
            'prediction': score_dict.get('prediction', False),
            'confidence': score_dict.get('confidence', '') if score_dict.get('prediction', False) else ''
        }
        return result
    except Exception as e:
        print(f"Error in watermark detection: {e}")
        return {
            'num_tokens_scored': 0,
            'num_green_tokens': 0,
            'green_fraction': 0.0,
            'z_score': 0.0,
            'p_value': 1.0,
            'prediction': False,
            'confidence': ''
        }

def process_human_texts_c4(dataset, tokenizer, detector, output_file="c4_human_texts_watermark_results.xlsx", 
                          max_samples=1000, target_human_tokens=200):
    """Process human texts from C4 dataset using the paper's method"""
    print(f"Processing C4 human texts... (max {max_samples} samples)")
    print(f"Using first 2 sentences as prompt, subsequent {target_human_tokens} tokens as human text")
    
    results = []
    processed_count = 0
    skipped_count = 0
    
    # Check if file already exists and ask user
    if os.path.exists(output_file):
        response = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping human text processing.")
            return
    
    try:
        pbar = tqdm(desc="Processing C4 human texts")
        
        for sample in dataset:
            if processed_count >= max_samples:
                break
                
            text = sample['text']
            
            # Extract prompt and human text using paper's method
            prompt, human_text = extract_prompt_and_human_text(text, tokenizer, target_human_tokens)
            
            if prompt is None or human_text is None:
                skipped_count += 1
                continue
            
            # Detect watermark in human text
            detection_result = detect_watermark(detector, human_text)
            
            # Combine results
            row = {
                'prompt': prompt,
                'human_text': human_text,
                'human_text_token_count': len(tokenizer.encode(human_text, add_special_tokens=False)),
                **detection_result
            }
            results.append(row)
            processed_count += 1
            
            pbar.set_postfix({
                'processed': processed_count,
                'skipped': skipped_count,
                'total_seen': processed_count + skipped_count
            })
            pbar.update(1)
            
        pbar.close()
        
        # Save to Excel
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)
        print(f"C4 human text results saved to {output_file}")
        print(f"Successfully processed: {len(results)} samples")
        print(f"Skipped (insufficient content): {skipped_count} samples")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            print(f"Partial results saved to {output_file} ({len(results)} samples)")

def generate_watermarked_texts_c4(dataset, tokenizer, model, watermark_processor, detector, 
                                 output_file="c4_model_generated_watermark_results.xlsx", 
                                 max_samples=1000, target_human_tokens=200, 
                                 min_new_tokens=150, max_new_tokens=250):
    """Generate watermarked texts using C4 prompts and detect watermarks"""
    print(f"Generating watermarked texts from C4 prompts... (max {max_samples} samples)")
    print(f"Target generation length: {min_new_tokens}-{max_new_tokens} tokens")
    
    results = []
    processed_count = 0
    skipped_count = 0
    
    # Check if file already exists and ask user
    if os.path.exists(output_file):
        response = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping text generation.")
            return
    
    try:
        pbar = tqdm(desc="Generating watermarked texts")
        
        for sample in dataset:
            if processed_count >= max_samples:
                break
                
            text = sample['text']
            
            # Extract prompt using the same method
            prompt, _ = extract_prompt_and_human_text(text, tokenizer, target_human_tokens)
            
            if prompt is None:
                skipped_count += 1
                continue
            
            try:
                # Tokenize input
                tokenized_input = tokenizer(prompt, return_tensors='pt').to(model.device)
                
                # Generate with watermark
                with torch.no_grad():
                    output_tokens = model.generate(
                        **tokenized_input,
                        logits_processor=LogitsProcessorList([watermark_processor]),
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Extract only newly generated tokens
                output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]
                generated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
                
                # Detect watermark in generated text
                detection_result = detect_watermark(detector, generated_text)
                
                # Combine results
                row = {
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'generated_text_token_count': len(tokenizer.encode(generated_text, add_special_tokens=False)),
                    **detection_result
                }
                results.append(row)
                processed_count += 1
                
                pbar.set_postfix({
                    'processed': processed_count,
                    'skipped': skipped_count,
                    'total_seen': processed_count + skipped_count
                })
                pbar.update(1)
                
            except Exception as e:
                print(f"Error generating for sample {processed_count + skipped_count}: {e}")
                skipped_count += 1
                continue
                
        pbar.close()
        
        # Save to Excel
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)
        print(f"C4 generated text results saved to {output_file}")
        print(f"Successfully processed: {len(results)} samples")
        print(f"Skipped (errors): {skipped_count} samples")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            print(f"Partial results saved to {output_file} ({len(results)} samples)")

def calculate_auc_from_results(human_file, generated_file, output_file="auc_results.txt"):
    """
    Calculate AUC from human and generated text results
    Args:
        human_file: Path to human text detection results Excel file
        generated_file: Path to generated text detection results Excel file
        output_file: Path to save AUC results
    """
    print("\n" + "="*60)
    print("CALCULATING AUC FROM Z-SCORES")
    print("="*60)
    
    try:
        # Read results files
        print(f"Reading human text results from: {human_file}")
        human_df = pd.read_excel(human_file)
        print(f"Reading generated text results from: {generated_file}")
        generated_df = pd.read_excel(generated_file)
        
        # Extract z_scores
        human_z_scores = human_df['z_score'].values
        generated_z_scores = generated_df['z_score'].values
        
        # Create labels (0 for human, 1 for generated)
        human_labels = np.zeros(len(human_z_scores))
        generated_labels = np.ones(len(generated_z_scores))
        
        # Combine data
        all_z_scores = np.concatenate([human_z_scores, generated_z_scores])
        all_labels = np.concatenate([human_labels, generated_labels])
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_z_scores)
        auc_score = auc(fpr, tpr)
        
        # Calculate statistics
        human_stats = {
            'count': len(human_z_scores),
            'mean': np.mean(human_z_scores),
            'std': np.std(human_z_scores),
            'min': np.min(human_z_scores),
            'max': np.max(human_z_scores)
        }
        
        generated_stats = {
            'count': len(generated_z_scores),
            'mean': np.mean(generated_z_scores),
            'std': np.std(generated_z_scores),
            'min': np.min(generated_z_scores),
            'max': np.max(generated_z_scores)
        }
        
        # Prepare results text
        results_text = f"""AUC Analysis Results
{'='*50}

AUC Score: {auc_score:.4f}

Human Text Z-Score Statistics:
- Count: {human_stats['count']}
- Mean: {human_stats['mean']:.4f}
- Std: {human_stats['std']:.4f}
- Min: {human_stats['min']:.4f}
- Max: {human_stats['max']:.4f}

Generated Text Z-Score Statistics:
- Count: {generated_stats['count']}
- Mean: {generated_stats['mean']:.4f}
- Std: {generated_stats['std']:.4f}
- Min: {generated_stats['min']:.4f}
- Max: {generated_stats['max']:.4f}

Performance Interpretation:
- AUC = 0.5: No discriminative ability (random)
- AUC > 0.7: Acceptable discrimination
- AUC > 0.8: Excellent discrimination
- AUC > 0.9: Outstanding discrimination

Current Performance: {'Outstanding' if auc_score > 0.9 else 'Excellent' if auc_score > 0.8 else 'Acceptable' if auc_score > 0.7 else 'Poor'}
"""
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results_text)
        
        # Print results
        print(results_text)
        print(f"AUC analysis results saved to: {output_file}")
        
        return auc_score, human_stats, generated_stats
        
    except Exception as e:
        error_msg = f"Error calculating AUC: {e}"
        print(error_msg)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"AUC Analysis Failed\n{'='*30}\n{error_msg}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='C4 Dataset Watermark Detection Experiment')
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-chat-hf", 
                       help='Model name for generation')
    parser.add_argument('--max_samples', type=int, default=300, 
                       help='Maximum number of samples to process')
    parser.add_argument('--target_human_tokens', type=int, default=200, 
                       help='Target number of tokens for human text extraction')
    parser.add_argument('--min_new_tokens', type=int, default=180, 
                       help='Minimum new tokens to generate')
    parser.add_argument('--max_new_tokens', type=int, default=220, 
                       help='Maximum new tokens to generate')
    parser.add_argument('--skip_human', action='store_true', 
                       help='Skip processing human texts')
    parser.add_argument('--skip_generation', action='store_true', 
                       help='Skip model generation')
    parser.add_argument('--human_output', default="", 
                       help='Output file for human text results')
    parser.add_argument('--generated_output', default="", 
                       help='Output file for generated text results')
    parser.add_argument('--auc_output', default="auc.txt", 
                       help='Output file for AUC analysis results')
    parser.add_argument('--subset', default="realnewslike", 
                       help='C4 subset to use (default: realnewslike)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.human_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.generated_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.auc_output), exist_ok=True)
    
    # Load C4 dataset
    print(f"Loading C4 dataset (subset: {args.subset})...")
    try:
        # Load the specified subset of C4
        dataset = load_dataset("c4", args.subset, split="train", streaming=True,trust_remote_code=True)
        print(f"C4 dataset loaded successfully (streaming mode)")
    except Exception as e:
        print(f"Error loading C4 dataset: {e}")
        print("Trying to load without subset specification...")
        dataset = load_dataset("c4", "en", split="train", streaming=True,trust_remote_code=True)
        print("C4 dataset loaded with 'en' subset")
    
    # Setup model and watermark components
    model, tokenizer, watermark_processor, watermark_detector = setup_model_and_watermark(args.model_name)
    
    # Process human texts (can be skipped)
    if not args.skip_human:
        print("\n" + "="*60)
        print("PROCESSING C4 HUMAN TEXTS (Paper's Method)")
        print("="*60)
        print("Method: First 2 sentences as prompt + subsequent 200 tokens as human text")
        process_human_texts_c4(
            dataset, 
            tokenizer,
            watermark_detector, 
            args.human_output, 
            args.max_samples,
            args.target_human_tokens
        )
    else:
        print("Skipping human text processing.")
    
    # Generate watermarked texts
    if not args.skip_generation:
        print("\n" + "="*60)
        print("GENERATING WATERMARKED TEXTS FROM C4 PROMPTS")
        print("="*60)
        
        # Reload dataset for generation (streaming dataset can only be iterated once)
        try:
            dataset = load_dataset("c4", args.subset, split="train", streaming=True)
        except:
            dataset = load_dataset("c4", "en", split="train", streaming=True)
        
        generate_watermarked_texts_c4(
            dataset, 
            tokenizer,
            model, 
            watermark_processor, 
            watermark_detector,
            args.generated_output,
            args.max_samples,
            args.target_human_tokens,
            args.min_new_tokens,
            args.max_new_tokens
        )
    else:
        print("Skipping model generation.")
    
    # Calculate AUC from results (always execute if both files should exist)
    if not args.skip_human and not args.skip_generation:
        calculate_auc_from_results(
            args.human_output,
            args.generated_output,
            args.auc_output
        )
    elif os.path.exists(args.human_output) and os.path.exists(args.generated_output):
        print("\nBoth result files exist, calculating AUC...")
        calculate_auc_from_results(
            args.human_output,
            args.generated_output,
            args.auc_output
        )
    else:
        print("\nSkipping AUC calculation - missing required result files.")
        print(f"Human results: {args.human_output} - {'Exists' if os.path.exists(args.human_output) else 'Missing'}")
        print(f"Generated results: {args.generated_output} - {'Exists' if os.path.exists(args.generated_output) else 'Missing'}")
    
    print("\nC4 experiment completed!")

if __name__ == "__main__":
    main()
