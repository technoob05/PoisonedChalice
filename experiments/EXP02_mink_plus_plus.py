
import argparse
import json
import random
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type, Any

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

# ============================================================================
# Model Loading
# ============================================================================

def load_model_from_directory(model_path: str):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=True, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False, trust_remote_code=True)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype, 
            device_map="auto"
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    model.eval()
    return model, tokenizer

# ============================================================================
# Min-K%++ Attack (Z-Score Normalization)
# ============================================================================

class MinKPlusPlusAttack:
    """
    Min-K%++ Attack:
    Normalizes token probabilities using the mean and variance of the model's 
    vocabulary distribution at each step.
    
    Score = Mean(Top-K% lowest Z-Scores)
    """
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.k = 0.2
        self.window_size = args.max_length if args.max_length != -1 else 256
        print(f"[Min-K++] Config: k={self.k}, window_size={self.window_size}")

    @property
    def name(self) -> str:
        return "mink_plus_plus"

    def calculate_z_scores(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, 
            max_length=self.window_size, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            # Log-Softmax over vocabulary
            log_probs = log_softmax(logits, dim=-1) # [1, seq_len, vocab_size]
            
            # Calculate Mean and Std over Vocabulary for each token position
            # This represents the "difficulty" of the prediction
            vocab_mean = log_probs.mean(dim=-1) # [1, seq_len]
            vocab_std = log_probs.std(dim=-1)   # [1, seq_len]
            
            # Z-Score Normalization
            # z = (x - mean) / std
            z_scores_dist = (log_probs - vocab_mean.unsqueeze(-1)) / (vocab_std.unsqueeze(-1) + 1e-8)
            
            # Extract Z-Scores for the Target Tokens
            target_z_scores = []
            for i in range(inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1]
                z_score = z_scores_dist[0, i, token_id].item()
                target_z_scores.append(z_score)

        return np.array(target_z_scores)

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"\nComputing {self.name} scores...")
        scores = []
        for text in tqdm(texts, desc="Z-Score Calculation"):
            try:
                # 1. Get Z-Scores of tokens
                z_scores = self.calculate_z_scores(text)
                
                # 2. Sort Z-Scores (descending or ascending?)
                # We want the "least likely" tokens (lowest probability/z-score)
                # Lower Z-score = More unexpected = Non-member behavior?
                # Wait. If member, the model should assign HIGH probability (High Z-score).
                # If non-member, low prob (Low Z-score).
                # Min-K checks the *minimum* probabilities. If the minimums are high, it's a member.
                
                sorted_z = np.sort(z_scores) # Ascending: [Low Z, ..., High Z]
                
                # Take top-k% lowest values
                k_len = max(1, int(len(sorted_z) * self.k))
                min_k_z = sorted_z[:k_len]
                
                # Average them
                score = np.mean(min_k_z)
                scores.append(score)
            except Exception as e:
                print(f"Error: {e}")
                scores.append(np.nan)
        return scores

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

        self.model, self.tokenizer = load_model_from_directory(args.model_name)

    def load_datasets(self) -> pd.DataFrame:
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"Loading dataset from: {self.args.dataset} (Local: {is_local})")
        for subset in subsets:
            if is_local:
                subset_path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(subset_path): continue
                ds = load_from_disk(subset_path)
                if hasattr(ds, "keys") and "test" in ds.keys(): ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            dfs.append(ds.to_pandas())
        ds = pd.concat(dfs, ignore_index=True)
        ds['is_member'] = ds['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        if self.args.sample_fraction < 1.0:
            ds = ds.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(ds)} examples")
        return ds

    def run(self):
        df = self.load_datasets()
        attacker = MinKPlusPlusAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        
        df[f"{attacker.name}_score"] = scores
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP02_{self.args.model_name.replace('/', '_')}_{timestamp}"
        output_file = self.output_dir / f"{exp_id}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"Saved to {output_file}")
        
        # AUC
        y_true = df['is_member']
        y_scores = df[f"{attacker.name}_score"].fillna(-999)
        try:
            auc = roc_auc_score(y_true, y_scores)
            print(f"AUC: {auc:.4f}")
        except:
            print("AUC computation failed (maybe all one class?)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AISE-TUDelft/Poisoned-Chalice")
    parser.add_argument("--sample_fraction", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    MIAExperiment(args).run()
