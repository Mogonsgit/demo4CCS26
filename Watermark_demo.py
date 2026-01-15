# coding=utf-8
from __future__ import annotations
import collections
import json
import pickle
from pathlib import Path
from math import sqrt, log
from typing import Dict

import scipy.stats
import numpy as np

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from normalizers import normalization_strategy_lookup


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


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
        bigram_table_path: str = None,
        entropy_skip_percentile: float = None,
        probability_aware_greenlist: bool = False,
        high_prob_top_p: float = 0.8,
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        
        # entropy-based skipping parameters
        self.bigram_table_path = bigram_table_path
        self.entropy_skip_percentile = entropy_skip_percentile
        self.bigram_table = None
        self.token_entropy_cache = {}
        self.entropy_threshold = None
        
        # 
        self.probability_aware_greenlist = probability_aware_greenlist
        self.high_prob_top_p = high_prob_top_p
        
        # 2-gram，
        if bigram_table_path is not None:
            self._load_and_compute_entropy()

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
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
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
                
                # GPUtoken
                high_prob_green_tokens = self._gpu_dp_select_green_tokens(
                    high_prob_tokens, 
                    high_prob_probs, 
                    self.gamma,
                    device
                )
                
                # mask
                vocab_mask = torch.ones(self.vocab_size, device=device, dtype=torch.bool)
                vocab_mask[high_prob_tokens] = False
                remaining_vocab = torch.nonzero(vocab_mask, as_tuple=False).squeeze(1)
                
                # token
                remaining_green_count = int(remaining_vocab.numel() * self.gamma)
                
                if remaining_green_count > 0 and remaining_vocab.numel() > 0:
                    # 
                    vocab_permutation = torch.randperm(remaining_vocab.numel(), device=device, generator=self.rng)
                    
                    if self.select_green_tokens:
                        selected_indices = vocab_permutation[:remaining_green_count]
                    else:
                        selected_indices = vocab_permutation[-remaining_green_count:]
                    
                    additional_green_tokens = remaining_vocab[selected_indices]
                    greenlist_ids = torch.cat([high_prob_green_tokens, additional_green_tokens])
                else:
                    greenlist_ids = high_prob_green_tokens
                
                return greenlist_ids
        
        # 
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        if self.select_green_tokens:  # directly
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
    
class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        # batch
        for b_idx in range(scores.shape[0]):
            # batchgreenlist
            greenlist_positions = torch.where(greenlist_mask[b_idx])[0]
            
            if len(greenlist_positions) == 0:
                continue
            
            # greenlistscores
            greenlist_scores = scores[b_idx][greenlist_positions]
            
            # 
            positive_mask = greenlist_scores >= 0
            if not positive_mask.any():
                continue
            
            positive_positions = greenlist_positions[positive_mask]
            positive_scores = greenlist_scores[positive_mask]
            
            # 40（40）
            top_k = min(40, len(positive_scores))
            top_values, top_indices = torch.topk(positive_scores, top_k)
            
            # 
            top_positions = positive_positions[top_indices]
            
            # bias
            scores[b_idx][top_positions] = scores[b_idx][top_positions] * (1 + greenlist_bias)
        
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            # token
            if input_ids.shape[1] > 0:  # token
                prev_token_id = input_ids[b_idx, -1].item()
                if self._should_skip_token(prev_token_id):
                    # ，scores
                    continue
            
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        # 
        non_skipped_indices = [i for i, gl_ids in enumerate(batched_greenlist_ids) if gl_ids is not None]
        
        if non_skipped_indices:
            non_skipped_greenlist_ids = [batched_greenlist_ids[i] for i in non_skipped_indices]
            non_skipped_scores = scores[non_skipped_indices]
            
            green_tokens_mask = self._calc_greenlist_mask(scores=non_skipped_scores, greenlist_token_ids=non_skipped_greenlist_ids)
            non_skipped_scores = self._bias_greenlist_logits(scores=non_skipped_scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
            
            # scores
            scores[non_skipped_indices] = non_skipped_scores
        
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
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

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
        return_skip_stats: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            
            # bigrams
            total_bigrams = len(freq.keys())
            skipped_bigrams = 0
            filtered_freq = {}
            for bigram, count in freq.items():
                if self._should_skip_token(bigram[0]):
                    skipped_bigrams += 1
                else:
                    filtered_freq[bigram] = count
            
            num_tokens_scored = len(filtered_freq.keys())
            num_tokens_skipped = skipped_bigrams
            
            for idx, bigram in enumerate(filtered_freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            # Standard method with entropy-based skipping
            scored_positions = []
            skipped_positions = []
            green_token_count, green_token_mask = 0, []
            
            for idx in range(self.min_prefix_len, len(input_ids)):
                prev_token = input_ids[idx - 1].item()
                
                # 
                if self._should_skip_token(prev_token):
                    skipped_positions.append(idx)
                    continue
                    
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                scored_positions.append(idx)
                
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
            
            num_tokens_scored = len(scored_positions)
            num_tokens_skipped = len(skipped_positions)
            
            if num_tokens_scored < 1:
                raise ValueError(
                    f"Must have at least 1 token to score after filtering and "
                    f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                )

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))
        if return_skip_stats:
            score_dict.update(dict(
                num_tokens_skipped=num_tokens_skipped,
                num_tokens_not_skipped=num_tokens_scored
            ))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict