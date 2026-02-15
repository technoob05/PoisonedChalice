
import argparse
import json
import random
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type, Any, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

# ============================================================================
# Model Loading
# ============================================================================

def load_model_from_directory(model_path: str):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,
            use_fast=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Fast tokenizer failed: {e}. Falling back to slow tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,
            use_fast=False,
            trust_remote_code=True,
        )

    # Configure model loading
    # Use float16 if available for memory efficiency (T4/P100 friendly)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model with device_map='auto': {e}")
        # Fallback for some models/environments
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.eval()
    return model, tokenizer

# ============================================================================
# Attack Base Class
# ============================================================================

class MIAttack(ABC):
    """Abstract base class for Membership Inference Attacks."""

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the attack."""
        pass

    @abstractmethod
    def compute_scores(self, texts: List[str]) -> List[float]:
        """
        Compute MIA scores for a list of texts.
        Returns: List of scores (higher score = more likely to be a member)
        """
        pass

# ============================================================================
# Loss Attack
# ============================================================================

class LossAttack(MIAttack):
    """Loss-based MIA: Lower loss indicates membership (so we return negative loss)."""

    @property
    def name(self) -> str:
        return "loss"

    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc=f"Computing {self.name} scores"):
            try:
                inputs = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                # Negative loss: higher score indicates membership
                scores.append(-outputs.loss.item())
            except Exception as e:
                print(f"Error calculating loss: {e}")
                scores.append(np.nan)
        return scores

# ============================================================================
# Min-K% Prob Attack
# ============================================================================

class MinKProbAttack(MIAttack):
    """
    Min-K% Prob Attack:
    Calculates the average log-probability of the k% tokens with the lowest probabilities.
    """
    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)
        self.sw = args.use_sliding_window
        self.k = 0.2  # Default k ratio
        self.window_size = args.max_length if args.max_length != -1 else 256
        print(f"[Min-K] Config: k={self.k}, window_size={self.window_size}")

    @property
    def name(self) -> str:
        return "mkp"

    def calculate_score_offline(self, sorted_probs: np.ndarray, k_ratio: float):
        if len(sorted_probs) == 0:
            return np.nan
        k_length = max(1, int(len(sorted_probs) * k_ratio))
        top_k_min_log_probs = sorted_probs[:k_length]
        return np.mean(top_k_min_log_probs)

    @staticmethod
    def calculate_token_probabilities_truncation(text: str, model, tokenizer, max_length) -> np.ndarray:
        inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            log_probs = log_softmax(logits, dim=-1)
            
            # Extract probs for target tokens (shifted by 1)
            # inputs specific indexing to get (batch, seq, vocab) -> (seq) values
            # logits: [1, seq_len, vocab_size]
            # labels: [1, seq_len]
            
            # Torch gather is usually faster/cleaner
            # Shift labels to allow for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate CrossEntropy (individual) -> equivalent to gathering log_probs
            # But we want log_probs specifically
            # Let's stick to the loop if it works, or optimize
            
            token_log_probs = []
            # Using the loop implementation from original code for fidelity
            for i in range(inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1]
                token_log_prob = log_probs[0, i, token_id].item()
                token_log_probs.append(token_log_prob)

        return np.array(token_log_probs)

    def get_token_probs(self, texts: List[str]) -> List[np.ndarray]:
        all_probs = []
        for text in tqdm(texts, desc="Calculating token probabilities for MKP"):
            # We strictly use truncation for now as sliding window is complex and slow
            probs = self.calculate_token_probabilities_truncation(
                text, self.model, self.tokenizer, self.window_size
            )
            all_probs.append(np.sort(probs))
        return all_probs

    def compute_scores(self, texts: List[str]) -> List[float]:
        all_sorted_probs = self.get_token_probs(texts)
        scores = []
        for probs in all_sorted_probs:
            scores.append(self.calculate_score_offline(probs, self.k))
        return scores

# ============================================================================
# PAC Attack
# ============================================================================

class PACAttack(MIAttack):
    """PAC-based MIA: Polarized-Augment Calibration"""

    @property
    def name(self) -> str:
        return "pac"
    
    @staticmethod
    def calculate_token_probabilities_sequential(texts: List[str], model, tokenizer, max_length: int = 4096) -> List[np.ndarray]:
        """Calculate per-token log probabilities for multiple texts sequentially."""
        all_token_log_probs = []
        for text in texts:
            try:
                inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    logits = outputs.logits
                    log_probs = log_softmax(logits, dim=-1)
                    
                    token_log_probs = []
                    seq_len = inputs["input_ids"].shape[1]
                    for i in range(seq_len - 1):
                        token_id = inputs["input_ids"][0, i + 1]
                        token_log_prob = log_probs[0, i, token_id].item()
                        token_log_probs.append(token_log_prob)
                    all_token_log_probs.append(np.array(token_log_probs))
            except Exception as e:
                print(f"PAC Error processing text: {e}")
                all_token_log_probs.append(np.array([]))
        return all_token_log_probs

    @staticmethod
    def compute_polarized_distance(list_of_probs: List[float], near_count: int, far_count: int) -> float:
        if len(list_of_probs) == 0: return 0.0
        
        list_length = len(list_of_probs)
        far_count = max(1, min(far_count, list_length))
        near_count = max(1, min(near_count, list_length))
        sorted_probs = np.sort(list_of_probs)
        
        # Scaling if counts exceed length
        if near_count + far_count > list_length:
            scale = list_length / (near_count + far_count)
            near_count = max(1, int(near_count * scale))
            far_count = max(1, int(far_count * scale))

        # Mean of highest (far) - Mean of lowest (near) ?? 
        # Original code: mean(sorted[::-1][:far]) - mean(sorted[:near])
        # sorted_probs[::-1] is descending (highest first). 
        # So it takes Top 'far' highest probs and Bottom 'near' lowest probs.
        return (
            np.mean(sorted_probs[::-1][:far_count]) -
            np.mean(sorted_probs[:near_count])
        )

    def generate_adjacent_samples(self, text: str, m_ratio: float = 0.3, n_samples: int = 5) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        adjacent_samples = []
        for _ in range(n_samples):
            swapped_tokens = tokens.copy()
            # Swap m_ratio of tokens
            num_swaps = int(m_ratio * len(swapped_tokens))
            for _ in range(num_swaps):
                if len(swapped_tokens) >= 2:
                    idx1, idx2 = random.sample(range(len(swapped_tokens)), 2)
                    swapped_tokens[idx1], swapped_tokens[idx2] = swapped_tokens[idx2], swapped_tokens[idx1]
            adjacent_samples.append(self.tokenizer.decode(swapped_tokens, skip_special_tokens=True))
        return adjacent_samples

    def compute_scores(self, texts: List[str]) -> List[float]:
        print("\nCalculating PAC scores (this may take a while)...")
        pac_scores = []
        
        for text in tqdm(texts, desc="PAC calculation"):
            try:
                # 1. Generate perturbations
                mutated_samples = self.generate_adjacent_samples(
                    text, self.args.pac_m_ratio, self.args.pac_n_samples
                )
                
                # 2. Get probs for original + mutations
                all_texts = [text] + mutated_samples
                all_probs = self.calculate_token_probabilities_sequential(
                    all_texts, self.model, self.tokenizer, self.args.max_length
                )
                
                original_probs = all_probs[0]
                mutated_probs_list = all_probs[1:]
                
                # 3. Compute Polarized Distances
                orig_pd = self.compute_polarized_distance(
                    original_probs.tolist(), self.args.pac_near_count, self.args.pac_far_count
                )
                
                mutated_pds = [
                    self.compute_polarized_distance(
                        mp.tolist(), self.args.pac_near_count, self.args.pac_far_count
                    ) for mp in mutated_probs_list
                ]
                
                # 4. Final Score: Original PD - Mean(Mutated PDs)
                if mutated_pds:
                    score = float(orig_pd) - float(np.mean(mutated_pds))
                else:
                    score = 0.0
                
                pac_scores.append(score)
                
            except Exception as e:
                print(f"Error calculating PAC: {e}")
                pac_scores.append(np.nan)
                
        return pac_scores

# ============================================================================
# Experiment Orchestrator
# ============================================================================

class MIAExperiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Initialize
        self.model, self.tokenizer = load_model_from_directory(args.model_name)
        
        # Registry
        self.attack_registry = {
            "loss": LossAttack,
            "pac": PACAttack,
            "mkp": MinKProbAttack,
        }

    def load_datasets(self) -> pd.DataFrame:
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        
        is_local = os.path.exists(self.args.dataset)
        print(f"Loading dataset from: {self.args.dataset} (Local: {is_local})")

        for subset in subsets:
            if is_local:
                subset_path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(subset_path):
                     # Try searching recursively or just skip if structure implies directly inside
                     # Sometimes datasets are just unzipped as subset folders
                     print(f"Warning: {subset_path} not found.")
                     continue
                ds = load_from_disk(subset_path)
                if hasattr(ds, "keys") and "test" in ds.keys():
                     ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
                
            dfs.append(ds.to_pandas())
            
        if not dfs:
            raise ValueError("No data loaded!")

        ds = pd.concat(dfs, ignore_index=True)
        ds['is_member'] = ds['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        # Sampling
        if self.args.sample_fraction < 1.0:
            ds = ds.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(ds)} examples ({self.args.sample_fraction*100}%)")
            
        return ds

    def save_results(self, df: pd.DataFrame, executed_attacks: List[str]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP01_{self.args.model_name.replace('/', '_')}_{timestamp}"
        output_file = self.output_dir / f"{exp_id}.parquet"

        df.to_parquet(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Calc and Print AUCs
        print("\n=== Final Results (AUC) ===")
        for attack in executed_attacks:
            score_col = f"{attack}_score"
            if score_col in df.columns:
                # Fill NaNs with min score to avoid error
                y_true = df['is_member']
                y_scores = df[score_col].fillna(df[score_col].min())
                try:
                    auc = roc_auc_score(y_true, y_scores)
                    print(f"Attack: {attack:10} | AUC: {auc:.4f}")
                except Exception as e:
                    print(f"Attack: {attack:10} | AUC: Error ({e})")
        print("===========================\n")

    def run(self):
        df = self.load_datasets()
        executed_attacks = []

        for attack_name in self.args.attacks:
            if attack_name not in self.attack_registry:
                print(f"Skipping unknown attack: {attack_name}")
                continue
            
            print(f"\nRunning attack: {attack_name}")
            attacker_cls = self.attack_registry[attack_name]
            attacker = attacker_cls(self.args, self.model, self.tokenizer)
            
            scores = attacker.compute_scores(df['content'].tolist())
            df[f"{attack_name}_score"] = scores
            executed_attacks.append(attack_name)

        self.save_results(df, executed_attacks)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AISE-TUDelft/Poisoned-Chalice")
    parser.add_argument("--attacks", nargs="+", default=["loss", "mkp", "pac"])
    parser.add_argument("--sample_fraction", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sliding_window", action="store_true")
    
    # PAC params
    parser.add_argument("--pac_near_count", type=int, default=30)
    parser.add_argument("--pac_far_count", type=int, default=5)
    parser.add_argument("--pac_m_ratio", type=float, default=0.3)
    parser.add_argument("--pac_n_samples", type=int, default=5)

    args = parser.parse_args()
    MIAExperiment(args).run()
