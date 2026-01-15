# coding=utf-8
from __future__ import annotations
import collections
import json
import pickle
from pathlib import Path
from math import sqrt, log
from typing import Dict, List, Tuple, Optional
import random

import scipy.stats
import numpy as np

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from normalizers import normalization_strategy_lookup
from RepeatCode import RepetitionEncoder, MajorityVotingDecoder


def load_bigram_table(file_path: str) -> Dict[int, Dict[int, float]]:
    """
    2-gram
    
    Args:
        file_path: ，.json.pkl
        
    Returns:
        2-gram， {prev_token: {next_token: probability}}
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"2-gram: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            json_table = json.load(f)
        # 
        table = {}
        for prev_token_str, next_dict in json_table.items():
            prev_token = int(prev_token_str)
            table[prev_token] = {}
            for next_token_str, prob in next_dict.items():
                next_token = int(next_token_str)
                table[prev_token][next_token] = float(prob)
        return table
    
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f": {file_path.suffix}，.json.pkl")


class HiddenMessageWatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        delta: float = 6.0,
        seeding_scheme: str = "simple_1",
        hash_key: int = 15485863,
        bigram_table_path: str = None,
        entropy_skip_percentile: float = None,
        message_length: int = 10,  # ：（）
        bits_per_char: int = 8,    # ：（8ASCII）
        probability_aware_greenlist: bool = False,
        high_prob_top_p: float = 0.8,
    ):
        # 
        self.vocab = vocab
        self.vocab_size = len(vocab) if vocab else 0
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        
        # 
        self.message_length = message_length
        self.bits_per_char = bits_per_char
        self.total_message_bits = message_length * bits_per_char
        
        # 
        self.bigram_table_path = bigram_table_path
        self.entropy_skip_percentile = entropy_skip_percentile
        self.bigram_table = None
        self.token_entropy_cache = {}
        self.entropy_threshold = None

        # 
        self.probability_aware_greenlist = probability_aware_greenlist
        self.high_prob_top_p = high_prob_top_p
        
        if bigram_table_path is not None:
            self._load_and_compute_entropy()
    
    def _encode_message(self, message: str) -> List[int]:
        """01"""
        if len(message) > self.message_length:
            raise ValueError(f" {len(message)}  {self.message_length}")
        
        # 
        message = message.ljust(self.message_length, '\0')  # 
        
        # ASCII
        bits = []
        for char in message:
            ascii_val = ord(char)
            # 
            for i in range(self.bits_per_char - 1, -1, -1):
                bits.append((ascii_val >> i) & 1)
        
        assert len(bits) == self.total_message_bits, f" {len(bits)}  {self.total_message_bits}"
        return bits
    
    def _decode_message(self, bits: List[int]) -> str:
        """01"""
        if len(bits) != self.total_message_bits:
            raise ValueError(f" {len(bits)}  {self.total_message_bits}")
        
        message = ""
        for i in range(0, len(bits), self.bits_per_char):
            byte = bits[i:i+self.bits_per_char]
            ascii_val = 0
            for bit in byte:
                ascii_val = (ascii_val << 1) | bit
            
            # （）
            if ascii_val == 0:
                break
            
            # 
            if 32 <= ascii_val <= 126:
                message += chr(ascii_val)
            elif ascii_val != 0:  # ，
                break
        
        return message
    
    def _load_and_compute_entropy(self):
        """2-gramtoken"""
        self.bigram_table = load_bigram_table(self.bigram_table_path)
        
        # token
        for prev_token, next_dict in self.bigram_table.items():
            entropy = 0.0
            for next_token, prob in next_dict.items():
                if prob > 0:
                    entropy -= prob * log(prob, 2)
            self.token_entropy_cache[prev_token] = entropy
        
        # ，
        if self.entropy_skip_percentile is not None:
            entropies = list(self.token_entropy_cache.values())
            self.entropy_threshold = np.percentile(entropies, self.entropy_skip_percentile)
    
    def _should_skip_token(self, token_id: int) -> bool:
        """token"""
        if self.entropy_threshold is None:
            return False
        
        token_entropy = self.token_entropy_cache.get(token_id, float('inf'))
        return token_entropy < self.entropy_threshold
    
    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme
        
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
    
    def _get_greenlist_ids(self, input_ids: torch.LongTensor, use_green: bool = True) -> torch.LongTensor:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids)
        
        # 2-gram
        if self.probability_aware_greenlist and self.bigram_table is not None:
            prev_token = input_ids[-1].item()
            
            # tokentoken
            if prev_token in self.bigram_table:
                next_token_probs = self.bigram_table[prev_token]
                
                # GPU tensor
                device = input_ids.device
                
                # token_idtensor
                token_ids = list(next_token_probs.keys())
                probs = list(next_token_probs.values())
                
                token_ids_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
                probs_tensor = torch.tensor(probs, device=device, dtype=torch.float32)
                
                # ，token
                sorted_indices = torch.argsort(probs_tensor, descending=True)
                sorted_token_ids = token_ids_tensor[sorted_indices]
                sorted_probs = probs_tensor[sorted_indices]
                
                # top_ptoken
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                mask = cumsum_probs <= self.high_prob_top_p
                
                # token，100
                num_tokens = min(max(1, mask.sum().item()), 100)
                high_prob_tokens = sorted_token_ids[:num_tokens]
                high_prob_probs = sorted_probs[:num_tokens]
                
                if use_green:
                    # ：GPUtoken
                    high_prob_green_tokens = self._gpu_dp_select_green_tokens(
                        high_prob_tokens, 
                        high_prob_probs, 
                        0.5,
                        device
                    )
                else:
                    # ：high_prob_tokens
                    # 
                    original_green_tokens = self._gpu_dp_select_green_tokens(
                        high_prob_tokens, 
                        high_prob_probs, 
                        0.5,
                        device
                    )
                    
                    # masktoken
                    high_prob_mask = torch.ones(high_prob_tokens.numel(), device=device, dtype=torch.bool)
                    for token in original_green_tokens:
                        token_indices = (high_prob_tokens == token).nonzero(as_tuple=True)[0]
                        if len(token_indices) > 0:
                            high_prob_mask[token_indices[0]] = False
                    
                    # tokentoken
                    high_prob_green_tokens = high_prob_tokens[high_prob_mask]
                
                # mask
                vocab_mask = torch.ones(self.vocab_size, device=device, dtype=torch.bool)
                vocab_mask[high_prob_tokens] = False
                remaining_vocab = torch.nonzero(vocab_mask, as_tuple=False).squeeze(1)
                
                # token
                remaining_green_count = int(remaining_vocab.numel() * 0.5)
                
                if remaining_green_count > 0 and remaining_vocab.numel() > 0:
                    # 
                    vocab_permutation = torch.randperm(remaining_vocab.numel(), device=device, generator=self.rng)
                    
                    if use_green:
                        selected_indices = vocab_permutation[:remaining_green_count]
                    else:
                        selected_indices = vocab_permutation[-remaining_green_count:]
                    
                    additional_green_tokens = remaining_vocab[selected_indices]
                    greenlist_ids = torch.cat([high_prob_green_tokens, additional_green_tokens])
                else:
                    greenlist_ids = high_prob_green_tokens
                
                return greenlist_ids
        
        # 
        greenlist_size = int(self.vocab_size * 0.5)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        if use_green:  # directly
            greenlist_ids = vocab_permutation[:greenlist_size]  # new
        else:  # select green via red
            greenlist_ids = vocab_permutation[-greenlist_size:]  # legacy behavior
        return greenlist_ids
        
    def _gpu_dp_select_green_tokens(self, tokens: torch.LongTensor, token_probs: torch.FloatTensor, gamma: float, device: torch.device) -> torch.LongTensor:
        """
        GPUtoken，tokengamma * total_prob
        """
        if tokens.numel() == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        
        # 
        total_prob = token_probs.sum()
        target_prob = gamma * total_prob
        
        # 
        scale_factor = 10000
        scaled_probs = (token_probs * scale_factor).long()
        scaled_target = int(target_prob * scale_factor)
        
        n = tokens.numel()
        
        # DP
        # ，
        prev_dp = torch.zeros(scaled_target + 1, device=device, dtype=torch.bool)
        curr_dp = torch.zeros(scaled_target + 1, device=device, dtype=torch.bool)
        prev_dp[0] = True
        
        # ，
        # parent[i][w] = (prev_w, selected) (i,w)itoken
        parent = torch.full((n, scaled_target + 1, 2), -1, device=device, dtype=torch.long)
        
        # dp
        for i in range(n):
            weight = scaled_probs[i]
            curr_dp.fill_(False)
            
            # DP
            curr_dp |= prev_dp  # itoken
            
            # itoken
            if weight <= scaled_target:
                # scatter
                valid_indices = torch.nonzero(prev_dp[:scaled_target + 1 - weight], as_tuple=False).squeeze(1)
                if valid_indices.numel() > 0:
                    target_indices = valid_indices + weight
                    curr_dp[target_indices] = True
                    
                    # 
                    parent[i, target_indices, 0] = valid_indices
                    parent[i, target_indices, 1] = 1
            
            # 
            not_selected_mask = prev_dp & (~curr_dp | (curr_dp & (parent[i, :, 1] != 1)))
            not_selected_indices = torch.nonzero(not_selected_mask, as_tuple=False).squeeze(1)
            if not_selected_indices.numel() > 0:
                parent[i, not_selected_indices, 0] = not_selected_indices
                parent[i, not_selected_indices, 1] = 0
            
            prev_dp, curr_dp = curr_dp, prev_dp
        
        # target
        reachable_weights = torch.nonzero(prev_dp, as_tuple=False).squeeze(1)
        if reachable_weights.numel() == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        
        # target
        valid_weights = reachable_weights[reachable_weights <= scaled_target]
        if valid_weights.numel() == 0:
            best_weight = reachable_weights[0]
        else:
            best_weight = valid_weights[-1]  # ，
        
        # token
        selected_mask = torch.zeros(n, device=device, dtype=torch.bool)
        w = best_weight.item()
        
        for i in range(n - 1, -1, -1):
            if parent[i, w, 1] == 1:  # itoken
                selected_mask[i] = True
                w = parent[i, w, 0].item()
        
        selected_tokens = tokens[selected_mask]
        return selected_tokens


class HiddenMessageWatermarkLogitsProcessor(HiddenMessageWatermarkBase, LogitsProcessor):
    def __init__(self, hidden_message: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 
        if len(hidden_message) > self.message_length:
            raise ValueError(f" {len(hidden_message)}  {self.message_length}")
        
        # 
        self.hidden_message = hidden_message
        self.message_bits = self._encode_message(hidden_message)
        # print(f"'{hidden_message}' {len(self.message_bits)} : {self.message_bits}")
        
        # RepetitionEncoder
        if self.message_bits:
            self.repetition_encoder = RepetitionEncoder(self.message_bits)
        else:
            self.repetition_encoder = None
        
        self.encoding_count = 0  # 
    
    def _get_next_encoded_bit(self) -> Optional[int]:
        """（）"""
        if self.repetition_encoder is None:
            return None
        
        try:
            bit = self.repetition_encoder.get_current_bit()
            self.encoding_count += 1
            # print(f"{self.encoding_count}，bit: {bit}")
            return bit
        except ValueError:
            return None
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask
    
    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] * (1 + greenlist_bias)
        return scores
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)
        
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
        
        for b_idx in range(input_ids.shape[0]):
            # token
            if input_ids.shape[1] > 0:
                prev_token_id = input_ids[b_idx, -1].item()
                if self._should_skip_token(prev_token_id):
                    continue
            
            # 
            bit = self._get_next_encoded_bit()
            
            # token
            if bit is None:
                # ，
                greenlist_ids = self._get_greenlist_ids(input_ids[b_idx], use_green=True)
            else:
                # 1，0
                greenlist_ids = self._get_greenlist_ids(input_ids[b_idx], use_green=(bit == 1))
            
            batched_greenlist_ids[b_idx] = greenlist_ids
        
        # 
        non_skipped_indices = [i for i, gl_ids in enumerate(batched_greenlist_ids) if gl_ids is not None]
        
        if non_skipped_indices:
            non_skipped_greenlist_ids = [batched_greenlist_ids[i] for i in non_skipped_indices]
            non_skipped_scores = scores[non_skipped_indices]
            
            green_tokens_mask = self._calc_greenlist_mask(scores=non_skipped_scores, greenlist_token_ids=non_skipped_greenlist_ids)
            non_skipped_scores = self._bias_greenlist_logits(scores=non_skipped_scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
            
            scores[non_skipped_indices] = non_skipped_scores
        
        return scores


class HiddenMessageWatermarkDetector(HiddenMessageWatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],
        ignore_repeated_bigrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"
        
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)
        
        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")
        
        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."
    
    def _extract_message_bits(self, input_ids: Tensor) -> Tuple[List[int], List[int]]:
        """token（01）"""
        extracted_bits = []
        bit_positions = []
        
        if self.ignore_repeated_bigrams:
            # bigram
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            
            for idx, bigram in enumerate(freq.keys()):
                if self._should_skip_token(bigram[0]):
                    continue
                
                prefix = torch.tensor([bigram[0]], device=self.device)
                
                # 
                greenlist_ids = self._get_greenlist_ids(prefix, use_green=True)
                
                # token
                actual_token = bigram[1]
                if actual_token in greenlist_ids:
                    extracted_bits.append(1)  # 1
                else:
                    extracted_bits.append(0)  # 0
                
                bit_positions.append(idx)
        else:
            # 
            for idx in range(self.min_prefix_len, len(input_ids)):
                prev_token = input_ids[idx - 1].item()
                
                if self._should_skip_token(prev_token):
                    continue
                
                curr_token = input_ids[idx].item()
                prefix = input_ids[:idx]
                
                # 
                greenlist_ids = self._get_greenlist_ids(prefix, use_green=True)
                # greenlist_ids_red = self._get_greenlist_ids(prefix, use_green=False)
                # if len(list(set(greenlist_ids)&set(greenlist_ids_red)))!=0:
                #     print(set(greenlist_ids)&set(greenlist_ids_red))
                
                # token
                if curr_token in greenlist_ids:
                    extracted_bits.append(1)  # 1
                else:
                    extracted_bits.append(0)  # 0
                
                bit_positions.append(idx)
        
        return extracted_bits, bit_positions
    
    def detect_hidden_message(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_debug_info: bool = False,
    ) -> dict:
        """（）"""
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        
        # 
        if text is not None:
            for normalizer in self.normalizers:
                text = normalizer(text)
        
        if tokenized_text is None:
            assert self.tokenizer is not None
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].to(self.device)
            # if tokenized_text[0] == self.tokenizer.bos_token_id:
            #     tokenized_text = tokenized_text[1:]
        else:
            tokenized_text = torch.tensor(tokenized_text, device=self.device)
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]
        
        # （01）
        extracted_bits, bit_positions = self._extract_message_bits(tokenized_text)
        
        if not extracted_bits:
            return {
                'detected': False,
                'message': None,
                'confidence': 0.0
            }
        
        # 
        if len(extracted_bits) < self.total_message_bits:
            return {
                'detected': False,
                'message': None,
                'confidence': 0.0,
                'error': f' {len(extracted_bits)}  {self.total_message_bits} '
            }
        
        # extracted_bitstotal_message_bits
        if len(extracted_bits) % self.total_message_bits != 0:
            # 
            trim_length = (len(extracted_bits) // self.total_message_bits) * self.total_message_bits
            trimmed_bits = extracted_bits[:trim_length]
        else:
            trimmed_bits = extracted_bits
        try:
            # MajorityVotingDecoder
            decoder = MajorityVotingDecoder(self.total_message_bits)
            decoded_bits = decoder.decode(trimmed_bits)
            
            if decoded_bits and len(decoded_bits) == self.total_message_bits:
                message = self._decode_message(decoded_bits)
                confidence = self._calculate_confidence(trimmed_bits, decoded_bits)
                
                result = {
                    'detected': True,
                    'message': message,
                    'confidence': confidence
                }
                
                if return_debug_info:
                    result['debug'] = {
                        'extracted_bits': extracted_bits,
                        'trimmed_bits': trimmed_bits,
                        'decoded_bits': decoded_bits,
                        'bit_positions': bit_positions,
                        'total_positions': len(bit_positions),
                        'expected_length': self.total_message_bits,
                        'actual_length': len(trimmed_bits)
                    }
                
                return result
            else:
                return {
                    'detected': False,
                    'message': None,
                    'confidence': 0.0,
                    'error': ''
                }
        
        except Exception as e:
            return {
                'detected': False,
                'message': None,
                'confidence': 0.0,
                'error': f': {str(e)}'
            }
    
    def _calculate_confidence(self, extracted_bits: List[int], decoded_bits: List[int]) -> float:
        """"""
        if not decoded_bits:
            return 0.0
        
        # RepetitionEncoder
        encoder = RepetitionEncoder(decoded_bits)
        re_encoded = []
        
        # ，extracted_bits
        for _ in range(len(extracted_bits)):
            try:
                re_encoded.append(encoder.get_current_bit())
            except ValueError:
                break
        
        # 
        matches = 0
        total = min(len(extracted_bits), len(re_encoded))
        
        for i in range(total):
            if extracted_bits[i] == re_encoded[i]:
                matches += 1
        
        if total == 0:
            return 0.0
        
        confidence = matches / total
        
        # ，0
        if confidence < 0.7:
            return 0.0
        
        return confidence
    
    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:
        """（）"""
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True
        
        # 
        if text is not None:
            for normalizer in self.normalizers:
                text = normalizer(text)
        
        if tokenized_text is None:
            assert self.tokenizer is not None
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            tokenized_text = torch.tensor(tokenized_text, device=self.device)
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]
        
        # 
        extracted_bits, _ = self._extract_message_bits(tokenized_text)
        
        if not extracted_bits:
            return {
                "num_tokens_scored": 0,
                "prediction": False,
                "confidence": 0.0
            }
        
        # token
        green_count = sum(1 for b in extracted_bits if b == 1)
        total_count = len(extracted_bits)
        green_fraction = green_count / total_count
        
        # z-score
        expected_fraction = 0.5  # 
        z_score = (green_fraction - expected_fraction) / sqrt(expected_fraction * (1 - expected_fraction) / total_count)
        p_value = scipy.stats.norm.sf(abs(z_score))
        
        output_dict = {
            "num_tokens_scored": total_count,
            "num_green_tokens": green_count,
            "green_fraction": green_fraction,
            "z_score": z_score,
            "p_value": p_value
        }
        
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            output_dict["prediction"] = abs(z_score) > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - p_value
        
        return output_dict