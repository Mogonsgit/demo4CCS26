#!/usr/bin/env python3
"""
2-gram
Wikipedia、Common Crawl2-gram
Hugging Face datasets
"""

import os
import json
import pickle
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Iterator
from pathlib import Path
import multiprocessing as mp
from functools import partial
import gc

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print(": tqdm，。: pip install tqdm")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print(": transformers，")

try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print(": datasets，Hugging Face")

class LargeCorpusBigramBuilder:
    """2-gram"""
    
    def __init__(
        self, 
        tokenizer_name: str = None,
        max_seq_length: int = 512,
        vocab_size_limit: Optional[int] = None,
        batch_size: int = 1000,
        save_interval: int = 100000,
    ):
        self.max_seq_length = max_seq_length
        self.vocab_size_limit = vocab_size_limit
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        # tokenizer
        if HAS_TRANSFORMERS and tokenizer_name:
            print(f"tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.vocab = list(range(len(self.tokenizer)))
            if vocab_size_limit:
                self.vocab = self.vocab[:vocab_size_limit]
        else:
            print("")
            self.tokenizer = None
            self.vocab = None
        
        # 
        self.bigram_counts = defaultdict(int)
        self.token_counts = defaultdict(int)
        self.processed_lines = 0
        self.word_to_id = {} if not self.tokenizer else None
        self.current_vocab_size = 0
    
    def _tokenize_text(self, text: str) -> List[int]:
        """"""
        if self.tokenizer:
            # transformers tokenizer
            tokens = self.tokenizer.encode(
                text, 
                add_special_tokens=False, 
                max_length=self.max_seq_length,
                truncation=True
            )
            # token
            if self.vocab_size_limit:
                tokens = [t for t in tokens if t < self.vocab_size_limit]
            return tokens
        else:
            # 
            words = text.lower().split()
            tokens = []
            for word in words:
                if word not in self.word_to_id:
                    if self.vocab_size_limit and len(self.word_to_id) >= self.vocab_size_limit:
                        continue  # 
                    self.word_to_id[word] = self.current_vocab_size
                    self.current_vocab_size += 1
                tokens.append(self.word_to_id[word])
            return tokens
    
    def _process_text_batch(self, texts: List[str]) -> Tuple[Dict, Dict]:
        """"""
        batch_bigram_counts = defaultdict(int)
        batch_token_counts = defaultdict(int)
        
        for text in texts:
            if not text or not text.strip():
                continue
                
            tokens = self._tokenize_text(text.strip())
            
            if len(tokens) < 2:
                continue
            
            # bigram
            for i in range(len(tokens) - 1):
                prev_token = tokens[i]
                next_token = tokens[i + 1]
                batch_bigram_counts[(prev_token, next_token)] += 1
                batch_token_counts[prev_token] += 1
        
        return batch_bigram_counts, batch_token_counts
    
    def process_file(self, file_path: str, max_lines: Optional[int] = None):
        """"""
        print(f": {file_path}")
        
        # 
        total_lines = None
        if HAS_TQDM:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines = sum(1 for _ in f)
                if max_lines and max_lines < total_lines:
                    total_lines = max_lines
            except:
                total_lines = None
        
        batch = []
        line_count = 0
        
        # 
        progress_bar = None
        if HAS_TQDM:
            progress_bar = tqdm(
                total=total_lines,
                desc="",
                unit="",
                unit_scale=True
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if max_lines and line_count >= max_lines:
                        break
                    
                    batch.append(line)
                    line_count += 1
                    
                    # 
                    if progress_bar:
                        progress_bar.update(1)
                    
                    if len(batch) >= self.batch_size:
                        self._process_and_update_batch(batch)
                        batch = []
                        
                        if not HAS_TQDM and line_count % self.save_interval == 0:
                            print(f" {line_count} ")
                        
                        gc.collect()  # 
                
                # 
                if batch:
                    self._process_and_update_batch(batch)
        
        finally:
            if progress_bar:
                progress_bar.close()
    
    def process_dataset(
        self, 
        dataset_name: str, 
        dataset_config: Optional[str] = None,
        text_field: str = "text",
        max_samples: Optional[int] = None,
        split: str = "train",
        streaming: bool = False
    ):
        """Hugging Face"""
        if not HAS_DATASETS:
            raise ImportError("datasets: pip install datasets")
        
        print(f": {dataset_name}")
        if dataset_config:
            print(f": {dataset_config}")
        
        # 
        try:
            if streaming:
                dataset = load_dataset(
                    dataset_name, 
                    dataset_config, 
                    split=split,
                    streaming=True
                )
            else:
                dataset = load_dataset(
                    dataset_name, 
                    dataset_config, 
                    split=split
                )
        except Exception as e:
            print(f": {e}")
            raise
        
        print(f"")
        
        # （）
        total_samples = None
        if not streaming:
            try:
                total_samples = len(dataset)
                if max_samples and max_samples < total_samples:
                    total_samples = max_samples
            except:
                total_samples = None
        elif max_samples:
            total_samples = max_samples
        
        # 
        progress_bar = None
        if HAS_TQDM:
            progress_bar = tqdm(
                total=total_samples,
                desc="",
                unit="",
                unit_scale=True
            )
        
        # 
        batch = []
        processed_samples = 0
        
        try:
            for sample in dataset:
                if max_samples and processed_samples >= max_samples:
                    break
                
                # 
                text = sample.get(text_field, "")
                if not text:
                    continue
                
                batch.append(text)
                processed_samples += 1
                
                # 
                if progress_bar:
                    progress_bar.update(1)
                
                # 
                if len(batch) >= self.batch_size:
                    self._process_and_update_batch(batch)
                    batch = []
                    
                    if not HAS_TQDM and processed_samples % self.save_interval == 0:
                        print(f" {processed_samples} ")
                    
                    gc.collect()
            
            # 
            if batch:
                self._process_and_update_batch(batch)
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        print(f"， {processed_samples} ")
    
    def process_wikipedia_dataset(
        self,
        date: str = "20220301",
        language: str = "en",
        max_samples: Optional[int] = None,
        streaming: bool = True
    ):
        """Wikipedia"""
        dataset_name = "wikipedia"
        dataset_config = f"{date}.{language}"
        
        self.process_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            text_field="text",
            max_samples=max_samples,
            split="train",
            streaming=streaming
        )
    
    def _process_and_update_batch(self, batch: List[str]):
        """"""
        batch_bigram_counts, batch_token_counts = self._process_text_batch(batch)
        
        # 
        for bigram, count in batch_bigram_counts.items():
            self.bigram_counts[bigram] += count
        
        for token, count in batch_token_counts.items():
            self.token_counts[token] += count
        
        self.processed_lines += len(batch)
    
    def process_directory(
        self, 
        directory: str, 
        file_pattern: str = "*.txt",
        max_files: Optional[int] = None
    ):
        """"""
        directory_path = Path(directory)
        files = list(directory_path.glob(file_pattern))
        
        if max_files:
            files = files[:max_files]
        
        print(f" {len(files)} ")
        
        # 
        file_progress = None
        if HAS_TQDM:
            file_progress = tqdm(
                total=len(files),
                desc="",
                unit="",
                position=0
            )
        
        try:
            for i, file_path in enumerate(files):
                if not HAS_TQDM:
                    print(f" {i+1}/{len(files)}: {file_path}")
                else:
                    file_progress.set_description(f": {file_path.name}")
                
                self.process_file(str(file_path))
                
                if file_progress:
                    file_progress.update(1)
        
        finally:
            if file_progress:
                file_progress.close()
    
    def build_probability_table(
        self, 
        min_bigram_count: int = 5,
        min_token_count: int = 10
    ) -> Dict[int, Dict[int, float]]:
        """"""
        print("...")
        print(f"bigram: {len(self.bigram_counts)}")
        print(f"token: {len(self.token_counts)}")
        
        prob_table = defaultdict(dict)
        
        # token
        print("token...")
        valid_tokens = set()
        
        if HAS_TQDM:
            token_iter = tqdm(
                self.token_counts.items(),
                desc="token",
                unit="token",
                unit_scale=True
            )
        else:
            token_iter = self.token_counts.items()
        
        for token, count in token_iter:
            if count >= min_token_count:
                valid_tokens.add(token)
        
        print(f"token: {len(valid_tokens)}")
        
        # 
        print("...")
        
        if HAS_TQDM:
            bigram_iter = tqdm(
                self.bigram_counts.items(),
                desc="",
                unit="bigram",
                unit_scale=True
            )
        else:
            bigram_iter = self.bigram_counts.items()
        
        for (prev_token, next_token), bigram_count in bigram_iter:
            if (bigram_count >= min_bigram_count and 
                prev_token in valid_tokens and 
                next_token in valid_tokens):
                
                prev_count = self.token_counts[prev_token]
                probability = bigram_count / prev_count
                prob_table[prev_token][next_token] = probability
        
        print(f"token: {len(prob_table)}")
        
        # 
        total_transitions = sum(len(next_dict) for next_dict in prob_table.values())
        avg_transitions = total_transitions / len(prob_table) if prob_table else 0
        
        print(f": {total_transitions}")
        print(f"token: {avg_transitions:.2f}")
        
        return dict(prob_table)
    
    def save_table(self, prob_table: dict, output_path: str, save_metadata: bool = True):
        """"""
        print(f": {output_path}")
        
        # 
        if output_path.endswith('.json'):
            # JSON
            json_table = {}
            for prev_token, next_dict in prob_table.items():
                json_table[str(prev_token)] = {str(k): v for k, v in next_dict.items()}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_table, f, indent=2)
        
        elif output_path.endswith('.pkl'):
            with open(output_path, 'wb') as f:
                pickle.dump(prob_table, f)
        
        # 
        if save_metadata:
            metadata = {
                'total_lines_processed': self.processed_lines,
                'total_bigrams': len(self.bigram_counts),
                'total_tokens': len(self.token_counts),
                'filtered_prev_tokens': len(prob_table),
                'total_transitions': sum(len(next_dict) for next_dict in prob_table.values()),
                'tokenizer_info': {
                    'type': type(self.tokenizer).__name__ if self.tokenizer else 'SimpleTokenizer',
                    'vocab_size': len(self.tokenizer) if self.tokenizer else self.current_vocab_size,
                    'vocab_size_limit': self.vocab_size_limit,
                }
            }
            
            metadata_path = output_path.replace('.json', '_metadata.json').replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        # （）
        if self.word_to_id:
            vocab_path = output_path.replace('.json', '_vocab.json').replace('.pkl', '_vocab.json')
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.word_to_id, f, indent=2)
        
        print("!")


def main():
    parser = argparse.ArgumentParser(description='2-gram')
    
    # 
    parser.add_argument('--input', '-i', help='')
    parser.add_argument('--output', '-o', required=True, help=' (.json.pkl)')
    
    # 
    parser.add_argument('--dataset', help='Hugging Face')
    parser.add_argument('--dataset-config', help='')
    parser.add_argument('--text-field', default='text', help='')
    parser.add_argument('--split', default='train', help='')
    parser.add_argument('--streaming', action='store_true', help='')
    
    # Wikipedia
    parser.add_argument('--wikipedia', action='store_true', help='Wikipedia')
    parser.add_argument('--wiki-date', default='20220301', help='Wikipedia')
    parser.add_argument('--wiki-lang', default='en', help='Wikipedia')
    
    # 
    parser.add_argument('--tokenizer', '-t', help='tokenizer ( gpt2, bert-base-uncased)')
    parser.add_argument('--max-lines', type=int, help='')
    parser.add_argument('--max-files', type=int, help='')
    parser.add_argument('--max-samples', type=int, help='')
    parser.add_argument('--vocab-size-limit', type=int, help='')
    parser.add_argument('--min-bigram-count', type=int, default=5, help='bigram')
    parser.add_argument('--min-token-count', type=int, default=10, help='token')
    parser.add_argument('--batch-size', type=int, default=1000, help='')
    
    args = parser.parse_args()
    
    # 
    builder = LargeCorpusBigramBuilder(
        tokenizer_name=args.tokenizer,
        vocab_size_limit=args.vocab_size_limit,
        batch_size=args.batch_size,
    )
    
    # 
    if args.wikipedia:
        # Wikipedia
        builder.process_wikipedia_dataset(
            date=args.wiki_date,
            language=args.wiki_lang,
            max_samples=args.max_samples,
            streaming=args.streaming
        )
    elif args.dataset:
        # Hugging Face
        builder.process_dataset(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            text_field=args.text_field,
            max_samples=args.max_samples,
            split=args.split,
            streaming=args.streaming
        )
    elif args.input:
        # 
        input_path = Path(args.input)
        if input_path.is_file():
            builder.process_file(str(input_path), max_lines=args.max_lines)
        elif input_path.is_dir():
            builder.process_directory(str(input_path), max_files=args.max_files)
        else:
            raise ValueError(f": {input_path}")
    else:
        raise ValueError(": --input, --dataset,  --wikipedia")
    
    # 
    prob_table = builder.build_probability_table(
        min_bigram_count=args.min_bigram_count,
        min_token_count=args.min_token_count
    )
    
    # 
    builder.save_table(prob_table, args.output)
    
    print("\n===  ===")
    print(f": {builder.processed_lines}")
    print(f"token: {len(prob_table)}")


def example_usage():
    """"""
    print("=== 2-gram ===\n")
    
    # Wikipedia
    print("1. Wikipedia:")
    print("python corpus_processor.py --wikipedia --output wiki_bigram_table.json --tokenizer gpt2")
    print("python corpus_processor.py --wikipedia --wiki-date 20220301 --wiki-lang en --max-samples 100000 --output wiki_bigram_table.pkl")
    print()
    
    # 
    print("2. Hugging Face:")
    print("python corpus_processor.py --dataset openwebtext --output owt_bigram_table.json --streaming")
    print("python corpus_processor.py --dataset c4 --dataset-config en --text-field text --output c4_bigram_table.pkl")
    print()
    
    # 
    print("3. :")
    print("python corpus_processor.py --input data.txt --output bigram_table.json --tokenizer gpt2")
    print("python corpus_processor.py --input ./corpus_dir --output bigram_table.pkl --max-files 100")
    print()
    
    # 
    print("4. :")
    print("python corpus_processor.py --wikipedia --output wiki_bigram_table.json \\")
    print("  --tokenizer gpt2 --vocab-size-limit 50000 --max-samples 1000000 \\")
    print("  --min-bigram-count 10 --min-token-count 50 --batch-size 2000")
    print()
    
    print("5. :")
    print("pip install datasets transformers torch tqdm")
    print()


def test_wikipedia_dataset():
    """Wikipedia"""
    print("=== Wikipedia ===\n")
    
    if not HAS_DATASETS:
        print(": datasets")
        print("pip install datasets")
        return
    
    # 
    builder = LargeCorpusBigramBuilder(
        vocab_size_limit=10000,  # 
        batch_size=50  # 
    )
    
    try:
        # Wikipedia
        print("Wikipedia...")
        builder.process_wikipedia_dataset(
            date="20220301",
            language="en", 
            max_samples=1000,  # 1000
            streaming=True
        )
        
        # 
        print("\n...")
        prob_table = builder.build_probability_table(
            min_bigram_count=2, 
            min_token_count=5
        )
        
        # 
        print("\n...")
        if HAS_TQDM:
            with tqdm(total=1, desc="") as pbar:
                builder.save_table(prob_table, "test_wiki_bigram_table.json")
                pbar.update(1)
        else:
            builder.save_table(prob_table, "test_wiki_bigram_table.json")
        
        print("\n===  ===")
        print(f": {builder.processed_lines}")
        print(f": {len(builder.word_to_id)}")
        print(f"bigram: {len(builder.bigram_counts)}")
        print(f"token: {len(prob_table)}")
        
        # bigram
        print("\n=== bigram ===")
        sorted_bigrams = sorted(
            builder.bigram_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        id_to_word = {v: k for k, v in builder.word_to_id.items()}
        
        for (prev_id, next_id), count in sorted_bigrams[:10]:
            prev_word = id_to_word.get(prev_id, f"UNK_{prev_id}")
            next_word = id_to_word.get(next_id, f"UNK_{next_id}")
            print(f"  '{prev_word}' -> '{next_word}': {count}")
        
        print("\n!")
        
    except Exception as e:
        print(f": {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        example_usage()
        print("\n:")
        print("python corpus_processor.py --test-wiki")
    elif "--test-wiki" in sys.argv:
        test_wikipedia_dataset()
    else:
        main()