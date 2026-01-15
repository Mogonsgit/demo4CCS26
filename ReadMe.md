# Watermark and Steganography Demo (Partial Release)

This repository contains **partial demonstration code** for the watermarking and steganography components of our paper. The full-featured implementations will be released **after the formal publication** of our work.

## 🔍 Files Overview

| Filename                   | Description                                                                     |
| -------------------------- | ------------------------------------------------------------------------------- |
| `test_watermark.py`        | Demonstration code for basic watermarking detection and generation.             |
| `test_hidden.py`           | Demonstration code for steganographic watermarking and hidden message recovery. |
| `Watermark_demo.py`        | Simplified version of our watermark generation and detection module.            |
| `Watermark_hidden_demo.py` | Simplified version of our hidden-message (steganography) watermark module.      |
| `RepeatCode.py`            | Simplified version of a repetition-based error-tolerant encoding scheme.        |
| `corpus_processor.py`      | Code to generate a probability-aware bigram table from a large corpus.          |
| `wiki_bigram_table.pkl`    | Precomputed probability-aware bigram table based on a Wikipedia dump.           |
| `experiments.py`        | Corpus-level evaluation script for testing watermark performance on the C4 dataset.     |

## ⚠️ Notice

The current implementation is **a reduced version** meant for demonstration only. It is designed to showcase the core concepts of our watermarking and hidden-message embedding techniques. The **complete version with full robustness and configurability** will be made available upon acceptance and publication of our paper.

## 🚀 Quick Start

### 1. Environment Setup

Make sure you have the following installed:

- Python 3.8+
- PyTorch with GPU support
- Huggingface `transformers` library
- Required models downloaded (e.g., `meta-llama/Llama-2-7b-chat-hf`)

You can install dependencies using:

```bash
pip install torch transformers
```

### 2. Run Basic Watermark Demo

```bash
python test_watermark.py
```

This will:
- Load a language model
- Generate a watermarked and a normal text from a prompt
- Detect watermark presence using statistical measures

### 3. Run Hidden Message Watermark Demo

```bash
python test_hidden.py
```

This will:
- Embed a short hidden string into generated text
- Detect and extract the embedded hidden message

### 4. Run C4 Corpus-Level Experiments
```bash
python experiments.py
```
The experiments.py script enables corpus-level evaluation on the C4 (Colossal Clean Crawled Corpus) and is intended for large-scale, realistic benchmarking of watermark behavior. In a typical run, the script performs the following steps:

- Loads a subset of the C4 dataset using the HuggingFace datasets library
- Samples raw text segments as generation prompts
- Generates continuations with and without watermarking under identical decoding settings
- Applies the watermark detector to each generated sample

## 📂 File Dependencies

Ensure that `wiki_bigram_table.pkl` is in the same directory, as both watermark modules rely on it for probability-aware operations.

If you want to regenerate the bigram table using your own corpus:

```bash
python corpus_processor.py
```

## 📌 Disclaimer

These scripts are shared for academic transparency and demonstration only. Unauthorized commercial use or distribution of the full methods is not permitted until the official release.

For questions, feel free to contact us.

---

© 2025 by the authors. All rights reserved.
