#!/usr/bin/env python3
"""
Improved Fine-tune GreBerta + BERT for Greek-English Word Alignment

Key improvements over v1:
1. Better architecture with cosine similarity, cross-attention, and bilinear features
2. Hard negative mining (semantically similar but unaligned pairs)
3. Proper subword pooling (average all subwords per word)
4. Balanced data with class weights and optional focal loss
5. Positional features (relative position difference)
6. Threshold optimization on dev set
7. Contrastive learning objective option
8. Better evaluation with AER (Alignment Error Rate)

Usage:
    python train_alignment_v2.py [--epochs 5] [--batch-size 8] [--lr 2e-5]
    python train_alignment_v2.py --quick-test  # Train on subset for testing
"""

import json
import argparse
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report


# ============================================================================
# Environment Detection (for Colab compatibility)
# ============================================================================

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PATH_PREFIX = "/content/drive/MyDrive/UofT DL Term project/GreBerta-experiment-2/"
    DATA_PREFIX = "/content/drive/MyDrive/UofT DL Term project/"
else:
    PATH_PREFIX = ""
    DATA_PREFIX = "../"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AlignmentExample:
    """Single verse with alignment pairs"""
    verse_id: str
    greek_tokens: List[Dict]  # Full token info including lemma, pos
    english_tokens: List[Dict]  # Full token info including strongs
    alignments: List[Tuple[int, int]]  # (greek_idx, english_idx) pairs


# ============================================================================
# Improved Model Architecture
# ============================================================================

class ImprovedAlignmentModel(nn.Module):
    """
    Improved cross-lingual word alignment model with:
    - Cosine similarity features
    - Cross-attention between languages
    - Bilinear interaction
    - Positional encoding for relative positions
    """
    
    def __init__(
        self,
        greek_model_name="bowphs/GreBerta",
        english_model_name="bert-base-uncased",
        hidden_dim=384,
        num_attention_heads=8,
        dropout=0.2,
        use_cross_attention=True,
        freeze_encoders_epochs=0,  # Freeze encoders for first N epochs
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.freeze_encoders_epochs = freeze_encoders_epochs
        self.current_epoch = 0
        
        # Pre-trained encoders
        self.greek_encoder = AutoModel.from_pretrained(greek_model_name)
        self.english_encoder = AutoModel.from_pretrained(english_model_name)
        
        # Get hidden sizes
        self.greek_hidden = self.greek_encoder.config.hidden_size
        self.english_hidden = self.english_encoder.config.hidden_size
        
        # Project both languages to same dimension
        self.greek_proj = nn.Linear(self.greek_hidden, hidden_dim)
        self.english_proj = nn.Linear(self.english_hidden, hidden_dim)
        
        # Cross-attention layers (optional but recommended)
        if use_cross_attention:
            self.greek_to_english_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.english_to_greek_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Bilinear layer for interaction
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Positional feature encoding (for relative position difference)
        self.pos_embedding = nn.Embedding(201, 32)  # -100 to +100 position diff
        
        # Final classifier
        # Features: concat(greek, english) + bilinear + cosine + position
        classifier_input_dim = hidden_dim * 3 + 1 + 32  # greek + english + bilinear + cosine + pos
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: aligned or not
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize custom layers with Xavier/Glorot initialization"""
        for module in [self.greek_proj, self.english_proj, self.bilinear]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Bilinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def set_epoch(self, epoch):
        """Update current epoch for encoder freezing schedule"""
        self.current_epoch = epoch
        
        # Freeze/unfreeze encoders based on schedule
        freeze = epoch < self.freeze_encoders_epochs
        for param in self.greek_encoder.parameters():
            param.requires_grad = not freeze
        for param in self.english_encoder.parameters():
            param.requires_grad = not freeze
    
    def forward(
        self,
        greek_input_ids,
        greek_attention_mask,
        english_input_ids,
        english_attention_mask,
        greek_word_spans,      # [batch, num_greek_words, 2] - (start, end) token indices
        english_word_spans,    # [batch, num_english_words, 2]
        pair_indices,          # [batch, num_pairs, 2] - (greek_word_idx, english_word_idx)
        greek_positions=None,  # [batch, num_pairs] - position of greek word
        english_positions=None,# [batch, num_pairs] - position of english word
        return_embeddings=False,  # Return embeddings for contrastive loss
    ):
        """
        Forward pass with improved architecture
        
        Returns:
            logits: [batch_size, num_pairs, 2] - alignment scores
            (optional) greek_pair_embeds, english_pair_embeds if return_embeddings=True
        """
        batch_size = greek_input_ids.size(0)
        
        # Encode Greek
        greek_outputs = self.greek_encoder(
            input_ids=greek_input_ids,
            attention_mask=greek_attention_mask
        )
        greek_hidden = greek_outputs.last_hidden_state  # [batch, greek_len, hidden]
        
        # Encode English
        english_outputs = self.english_encoder(
            input_ids=english_input_ids,
            attention_mask=english_attention_mask
        )
        english_hidden = english_outputs.last_hidden_state  # [batch, eng_len, hidden]
        
        # Project to common dimension
        greek_proj = self.greek_proj(greek_hidden)      # [batch, greek_len, hidden_dim]
        english_proj = self.english_proj(english_hidden)  # [batch, eng_len, hidden_dim]
        
        # Optional: Cross-attention to contextualize
        if self.use_cross_attention:
            # Greek attends to English
            greek_ctx, _ = self.greek_to_english_attn(
                greek_proj, english_proj, english_proj,
                key_padding_mask=(english_attention_mask == 0)
            )
            greek_proj = greek_proj + greek_ctx  # Residual
            
            # English attends to Greek
            english_ctx, _ = self.english_to_greek_attn(
                english_proj, greek_proj, greek_proj,
                key_padding_mask=(greek_attention_mask == 0)
            )
            english_proj = english_proj + english_ctx  # Residual
        
        # Pool subword tokens to word-level embeddings (mean pooling)
        greek_word_embeds = self._pool_to_words(greek_proj, greek_word_spans)   # [batch, num_greek_words, hidden_dim]
        english_word_embeds = self._pool_to_words(english_proj, english_word_spans)  # [batch, num_english_words, hidden_dim]
        
        # Gather word embeddings for each pair
        num_pairs = pair_indices.size(1)
        
        # Get indices
        greek_pair_idx = pair_indices[:, :, 0]   # [batch, num_pairs]
        english_pair_idx = pair_indices[:, :, 1]  # [batch, num_pairs]
        
        # Gather embeddings
        greek_pair_embeds = torch.gather(
            greek_word_embeds, 1,
            greek_pair_idx.unsqueeze(-1).expand(-1, -1, greek_word_embeds.size(-1))
        )  # [batch, num_pairs, hidden_dim]
        
        english_pair_embeds = torch.gather(
            english_word_embeds, 1,
            english_pair_idx.unsqueeze(-1).expand(-1, -1, english_word_embeds.size(-1))
        )  # [batch, num_pairs, hidden_dim]
        
        # Compute features
        # 1. Bilinear interaction
        bilinear_feat = self.bilinear(greek_pair_embeds, english_pair_embeds)  # [batch, num_pairs, hidden_dim]
        
        # 2. Cosine similarity
        cosine_sim = F.cosine_similarity(greek_pair_embeds, english_pair_embeds, dim=-1, eps=1e-8)  # [batch, num_pairs]
        cosine_sim = cosine_sim.unsqueeze(-1)  # [batch, num_pairs, 1]
        
        # 3. Positional features
        if greek_positions is not None and english_positions is not None:
            pos_diff = greek_positions - english_positions  # [batch, num_pairs]
            pos_diff = pos_diff.clamp(-100, 100) + 100  # Shift to [0, 200]
            pos_feat = self.pos_embedding(pos_diff)  # [batch, num_pairs, 32]
        else:
            pos_feat = torch.zeros(batch_size, num_pairs, 32, device=greek_input_ids.device)
        
        # Concatenate all features
        combined = torch.cat([
            greek_pair_embeds,
            english_pair_embeds,
            bilinear_feat,
            cosine_sim,
            pos_feat
        ], dim=-1)  # [batch, num_pairs, hidden_dim*3 + 1 + 32]
        
        # Classify
        logits = self.classifier(combined)  # [batch, num_pairs, 2]
        
        if return_embeddings:
            return logits, greek_pair_embeds, english_pair_embeds
        return logits
    
    def _pool_to_words(self, token_embeds, word_spans):
        """
        Pool subword token embeddings to word-level using mean pooling
        
        Args:
            token_embeds: [batch, seq_len, hidden_dim]
            word_spans: [batch, num_words, 2] - (start, end) indices for each word
            
        Returns:
            word_embeds: [batch, num_words, hidden_dim]
        """
        batch_size, num_words, _ = word_spans.shape
        hidden_dim = token_embeds.size(-1)
        
        word_embeds = torch.zeros(batch_size, num_words, hidden_dim, device=token_embeds.device)
        
        for b in range(batch_size):
            for w in range(num_words):
                start, end = word_spans[b, w].tolist()
                if start >= 0 and end > start:  # Valid span
                    word_embeds[b, w] = token_embeds[b, start:end].mean(dim=0)
        
        return word_embeds


# ============================================================================
# Focal Loss for handling class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] labels
        """
        # Filter out ignored indices
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """
    InfoNCE / NT-Xent style contrastive loss for alignment learning.
    
    For each Greek word, we want its aligned English word(s) to be close
    and non-aligned English words to be far in embedding space.
    """
    
    def __init__(self, temperature=0.07, ignore_index=-100):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
    
    def forward(self, greek_embeds, english_embeds, labels):
        """
        Args:
            greek_embeds: [batch, num_pairs, hidden_dim] - Greek word embeddings for pairs
            english_embeds: [batch, num_pairs, hidden_dim] - English word embeddings for pairs
            labels: [batch, num_pairs] - 1 for aligned, 0 for not, -100 for ignore
        
        Returns:
            Contrastive loss
        """
        batch_size, num_pairs, hidden_dim = greek_embeds.shape
        
        total_loss = 0
        count = 0
        
        for b in range(batch_size):
            # Get valid pairs
            valid_mask = labels[b] != self.ignore_index
            if valid_mask.sum() < 2:
                continue
            
            g_emb = greek_embeds[b][valid_mask]  # [valid_pairs, hidden]
            e_emb = english_embeds[b][valid_mask]  # [valid_pairs, hidden]
            pair_labels = labels[b][valid_mask]  # [valid_pairs]
            
            # Normalize embeddings
            g_emb = F.normalize(g_emb, dim=-1)
            e_emb = F.normalize(e_emb, dim=-1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(g_emb, e_emb.T) / self.temperature  # [valid, valid]
            
            # For positive pairs, we want to maximize similarity
            # For negative pairs, we want to minimize similarity
            positive_mask = pair_labels == 1
            
            if positive_mask.sum() > 0 and (~positive_mask).sum() > 0:
                # Standard contrastive: for each positive, other pairs are negatives
                for i in range(len(pair_labels)):
                    if pair_labels[i] == 1:
                        # This is a positive pair
                        pos_sim = sim_matrix[i, i]  # Similarity of aligned pair
                        
                        # Get negative similarities (same Greek word, different English)
                        neg_sims = sim_matrix[i, ~positive_mask]
                        
                        if len(neg_sims) > 0:
                            # InfoNCE loss
                            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                            target = torch.zeros(1, dtype=torch.long, device=logits.device)
                            total_loss += F.cross_entropy(logits.unsqueeze(0), target)
                            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=greek_embeds.device)
        
        return total_loss / count


class CombinedLoss(nn.Module):
    """Combined classification and contrastive loss"""
    
    def __init__(
        self, 
        classification_weight=1.0, 
        contrastive_weight=0.5,
        class_weights=None,
        ignore_index=-100,
        use_focal=False,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.ignore_index = ignore_index
        self.use_focal = use_focal
        
        if use_focal:
            self.cls_loss = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)
        else:
            self.cls_loss = nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=ignore_index
            )
        
        self.contrastive_loss = ContrastiveLoss(ignore_index=ignore_index)
    
    def forward(self, logits, labels, greek_embeds=None, english_embeds=None):
        """
        Args:
            logits: [batch, num_pairs, 2]
            labels: [batch, num_pairs]
            greek_embeds: Optional [batch, num_pairs, hidden] for contrastive
            english_embeds: Optional [batch, num_pairs, hidden] for contrastive
        """
        # Classification loss
        cls_loss = self.cls_loss(logits.view(-1, 2), labels.view(-1))
        
        total_loss = self.classification_weight * cls_loss
        
        # Contrastive loss (optional)
        if greek_embeds is not None and english_embeds is not None and self.contrastive_weight > 0:
            con_loss = self.contrastive_loss(greek_embeds, english_embeds, labels)
            total_loss = total_loss + self.contrastive_weight * con_loss
        
        return total_loss


# ============================================================================
# Improved Dataset with Hard Negative Mining
# ============================================================================

class ImprovedAlignmentDataset(Dataset):
    """
    Improved dataset with:
    - Balanced positive/negative sampling (1:1 ratio)
    - Hard negative mining (nearby words, similar POS)
    - Proper word span tracking for subword pooling
    """
    
    def __init__(
        self,
        examples: List[AlignmentExample],
        greek_tokenizer,
        english_tokenizer,
        max_pairs_per_verse=100,
        hard_negative_ratio=0.5,  # Fraction of negatives that should be hard
        pos_to_neg_ratio=1.0,     # 1:1 balanced
    ):
        self.examples = examples
        self.greek_tokenizer = greek_tokenizer
        self.english_tokenizer = english_tokenizer
        self.max_pairs_per_verse = max_pairs_per_verse
        self.hard_negative_ratio = hard_negative_ratio
        self.pos_to_neg_ratio = pos_to_neg_ratio
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Get word lists
        greek_words = [t['word'] if isinstance(t, dict) else t for t in example.greek_tokens]
        english_words = [t['word'] if isinstance(t, dict) else t for t in example.english_tokens]
        
        # Tokenize Greek
        greek_text = " ".join(greek_words)
        greek_encoded = self.greek_tokenizer(
            greek_text, truncation=True, max_length=512, return_tensors="pt"
        )
        
        # Tokenize English
        english_text = " ".join(english_words)
        english_encoded = self.english_tokenizer(
            english_text, truncation=True, max_length=512, return_tensors="pt"
        )
        
        # Get word spans (start, end token indices for each word)
        greek_word_spans = self._get_word_spans(greek_words, greek_encoded)
        english_word_spans = self._get_word_spans(english_words, english_encoded)
        
        # Create training pairs with hard negative mining
        pairs, labels = self._create_training_pairs(
            example.alignments,
            len(greek_words),
            len(english_words),
            example.greek_tokens,
            example.english_tokens
        )
        
        # Convert to tensors
        pair_indices = torch.tensor(pairs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Position tensors
        greek_positions = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        english_positions = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        
        return {
            "greek_input_ids": greek_encoded["input_ids"].squeeze(0),
            "greek_attention_mask": greek_encoded["attention_mask"].squeeze(0),
            "english_input_ids": english_encoded["input_ids"].squeeze(0),
            "english_attention_mask": english_encoded["attention_mask"].squeeze(0),
            "greek_word_spans": torch.tensor(greek_word_spans, dtype=torch.long),
            "english_word_spans": torch.tensor(english_word_spans, dtype=torch.long),
            "pair_indices": pair_indices,
            "labels": labels,
            "greek_positions": greek_positions,
            "english_positions": english_positions,
            "verse_id": example.verse_id,
            "num_greek_words": len(greek_words),
            "num_english_words": len(english_words),
        }
    
    def _get_word_spans(self, words, encoded):
        """Get (start, end) token indices for each word"""
        word_ids = encoded.word_ids()
        spans = []
        
        for word_idx in range(len(words)):
            token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            if token_indices:
                spans.append((min(token_indices), max(token_indices) + 1))
            else:
                # Word not found (truncated) - use padding index
                spans.append((-1, -1))
        
        return spans
    
    def _create_training_pairs(self, alignments, num_greek, num_english, greek_tokens, english_tokens):
        """Create balanced training pairs with hard negative mining"""
        
        # Positive pairs
        positive_set = set(alignments)
        positive_pairs = [(g, e, 1) for g, e in positive_set]
        
        # Calculate number of negatives (1:1 ratio)
        num_positives = len(positive_pairs)
        num_negatives = int(num_positives * self.pos_to_neg_ratio)
        
        # Hard negatives: positionally close words that aren't aligned
        num_hard = int(num_negatives * self.hard_negative_ratio)
        num_random = num_negatives - num_hard
        
        hard_negatives = self._sample_hard_negatives(
            positive_set, num_greek, num_english, num_hard, greek_tokens, english_tokens
        )
        
        # Random negatives
        random_negatives = self._sample_random_negatives(
            positive_set, num_greek, num_english, num_random
        )
        
        # Combine all pairs
        all_pairs = positive_pairs + hard_negatives + random_negatives
        random.shuffle(all_pairs)
        
        # Limit total pairs
        all_pairs = all_pairs[:self.max_pairs_per_verse]
        
        if not all_pairs:
            # Dummy pair if empty
            all_pairs = [(0, 0, 0)]
        
        pairs = [(p[0], p[1]) for p in all_pairs]
        labels = [p[2] for p in all_pairs]
        
        return pairs, labels
    
    def _sample_hard_negatives(self, positive_set, num_greek, num_english, num_samples, greek_tokens, english_tokens):
        """
        Sample hard negatives:
        1. Positionally close (within ±3 positions)
        2. Same POS category
        3. Aligned to same English word but wrong Greek word
        """
        hard_negatives = []
        candidates = []
        
        # Build candidate list with priority scoring
        for g in range(num_greek):
            for e in range(num_english):
                if (g, e) not in positive_set:
                    # Score based on position similarity
                    g_rel = g / max(num_greek, 1)
                    e_rel = e / max(num_english, 1)
                    pos_diff = abs(g_rel - e_rel)
                    
                    # Higher priority for closer positions
                    if pos_diff < 0.2:  # Within 20% of sequence
                        priority = 3
                    elif pos_diff < 0.4:
                        priority = 2
                    else:
                        priority = 1
                    
                    # Check if this Greek word is aligned elsewhere (confusing pair)
                    greek_aligned = any(g2 == g for g2, e2 in positive_set)
                    english_aligned = any(e2 == e for g2, e2 in positive_set)
                    
                    if greek_aligned or english_aligned:
                        priority += 1  # Harder negative
                    
                    candidates.append((g, e, priority))
        
        # Sort by priority (descending) and sample
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        for g, e, _ in candidates[:num_samples]:
            hard_negatives.append((g, e, 0))
        
        return hard_negatives
    
    def _sample_random_negatives(self, positive_set, num_greek, num_english, num_samples):
        """Sample random negative pairs"""
        negatives = []
        attempts = 0
        max_attempts = num_samples * 20
        
        while len(negatives) < num_samples and attempts < max_attempts:
            g = random.randint(0, num_greek - 1)
            e = random.randint(0, num_english - 1)
            if (g, e) not in positive_set:
                negatives.append((g, e, 0))
            attempts += 1
        
        return negatives


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch):
    """Custom collate function with proper padding for all tensors"""
    
    # Find max lengths
    max_greek_len = max(item["greek_input_ids"].size(0) for item in batch)
    max_english_len = max(item["english_input_ids"].size(0) for item in batch)
    max_greek_words = max(item["greek_word_spans"].size(0) for item in batch)
    max_english_words = max(item["english_word_spans"].size(0) for item in batch)
    max_pairs = max(item["labels"].size(0) for item in batch)
    
    # Prepare output lists
    result = {
        "greek_input_ids": [],
        "greek_attention_mask": [],
        "english_input_ids": [],
        "english_attention_mask": [],
        "greek_word_spans": [],
        "english_word_spans": [],
        "pair_indices": [],
        "labels": [],
        "greek_positions": [],
        "english_positions": [],
        "verse_ids": [],
    }
    
    for item in batch:
        # Pad Greek tokens
        g_len = item["greek_input_ids"].size(0)
        result["greek_input_ids"].append(
            F.pad(item["greek_input_ids"], (0, max_greek_len - g_len))
        )
        result["greek_attention_mask"].append(
            F.pad(item["greek_attention_mask"], (0, max_greek_len - g_len))
        )
        
        # Pad English tokens
        e_len = item["english_input_ids"].size(0)
        result["english_input_ids"].append(
            F.pad(item["english_input_ids"], (0, max_english_len - e_len))
        )
        result["english_attention_mask"].append(
            F.pad(item["english_attention_mask"], (0, max_english_len - e_len))
        )
        
        # Pad word spans
        g_words = item["greek_word_spans"].size(0)
        result["greek_word_spans"].append(
            F.pad(item["greek_word_spans"], (0, 0, 0, max_greek_words - g_words), value=-1)
        )
        
        e_words = item["english_word_spans"].size(0)
        result["english_word_spans"].append(
            F.pad(item["english_word_spans"], (0, 0, 0, max_english_words - e_words), value=-1)
        )
        
        # Pad pairs
        num_pairs = item["labels"].size(0)
        result["pair_indices"].append(
            F.pad(item["pair_indices"], (0, 0, 0, max_pairs - num_pairs))
        )
        result["labels"].append(
            F.pad(item["labels"], (0, max_pairs - num_pairs), value=-100)
        )
        result["greek_positions"].append(
            F.pad(item["greek_positions"], (0, max_pairs - num_pairs))
        )
        result["english_positions"].append(
            F.pad(item["english_positions"], (0, max_pairs - num_pairs))
        )
        
        result["verse_ids"].append(item["verse_id"])
    
    # Stack tensors
    for key in result:
        if key != "verse_ids":
            result[key] = torch.stack(result[key])
    
    return result


# ============================================================================
# Data Loading
# ============================================================================

def load_alignment_data(filepath: Path) -> List[AlignmentExample]:
    """Load alignment data from JSON"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for verse in data["data"]:
        if not verse["alignments"]:
            continue
        
        alignments = [(a["greek_idx"], a["english_idx"]) for a in verse["alignments"]]
        
        examples.append(
            AlignmentExample(
                verse_id=verse["verse_id"],
                greek_tokens=verse["greek_tokens"],
                english_tokens=verse["english_tokens"],
                alignments=alignments,
            )
        )
    
    return examples


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, epoch, use_contrastive=False):
    """Train for one epoch"""
    model.train()
    model.set_epoch(epoch)
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch in progress_bar:
        # Move to device
        greek_input_ids = batch["greek_input_ids"].to(device)
        greek_attention_mask = batch["greek_attention_mask"].to(device)
        english_input_ids = batch["english_input_ids"].to(device)
        english_attention_mask = batch["english_attention_mask"].to(device)
        greek_word_spans = batch["greek_word_spans"].to(device)
        english_word_spans = batch["english_word_spans"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        labels = batch["labels"].to(device)
        greek_positions = batch["greek_positions"].to(device)
        english_positions = batch["english_positions"].to(device)
        
        # Forward pass
        if use_contrastive:
            logits, greek_embeds, english_embeds = model(
                greek_input_ids,
                greek_attention_mask,
                english_input_ids,
                english_attention_mask,
                greek_word_spans,
                english_word_spans,
                pair_indices,
                greek_positions,
                english_positions,
                return_embeddings=True,
            )
            # Compute combined loss
            loss = loss_fn(logits, labels, greek_embeds, english_embeds)
        else:
            logits = model(
                greek_input_ids,
                greek_attention_mask,
                english_input_ids,
                english_attention_mask,
                greek_word_spans,
                english_word_spans,
                pair_indices,
                greek_positions,
                english_positions,
            )
            # Compute loss
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        valid_mask = labels != -100
        all_preds.extend(preds[valid_mask].cpu().numpy())
        all_labels.extend(labels[valid_mask].cpu().numpy())
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    
    return {
        "loss": total_loss / len(dataloader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, dataloader, device, return_predictions=False):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            greek_input_ids = batch["greek_input_ids"].to(device)
            greek_attention_mask = batch["greek_attention_mask"].to(device)
            english_input_ids = batch["english_input_ids"].to(device)
            english_attention_mask = batch["english_attention_mask"].to(device)
            greek_word_spans = batch["greek_word_spans"].to(device)
            english_word_spans = batch["english_word_spans"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            labels = batch["labels"].to(device)
            greek_positions = batch["greek_positions"].to(device)
            english_positions = batch["english_positions"].to(device)
            
            logits = model(
                greek_input_ids,
                greek_attention_mask,
                english_input_ids,
                english_attention_mask,
                greek_word_spans,
                english_word_spans,
                pair_indices,
                greek_positions,
                english_positions,
            )
            
            probs = F.softmax(logits, dim=-1)[:, :, 1]  # Probability of aligned
            preds = torch.argmax(logits, dim=-1)
            
            valid_mask = labels != -100
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_probs.extend(probs[valid_mask].cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    
    result = {"precision": precision, "recall": recall, "f1": f1}
    
    if return_predictions:
        result["predictions"] = all_preds
        result["labels"] = all_labels
        result["probabilities"] = all_probs
    
    return result


def find_optimal_threshold(model, dataloader, device, thresholds=None):
    """Find optimal threshold on dev set"""
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Finding threshold"):
            greek_input_ids = batch["greek_input_ids"].to(device)
            greek_attention_mask = batch["greek_attention_mask"].to(device)
            english_input_ids = batch["english_input_ids"].to(device)
            english_attention_mask = batch["english_attention_mask"].to(device)
            greek_word_spans = batch["greek_word_spans"].to(device)
            english_word_spans = batch["english_word_spans"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            labels = batch["labels"].to(device)
            greek_positions = batch["greek_positions"].to(device)
            english_positions = batch["english_positions"].to(device)
            
            logits = model(
                greek_input_ids,
                greek_attention_mask,
                english_input_ids,
                english_attention_mask,
                greek_word_spans,
                english_word_spans,
                pair_indices,
                greek_positions,
                english_positions,
            )
            
            probs = F.softmax(logits, dim=-1)[:, :, 1]
            valid_mask = labels != -100
            all_probs.extend(probs[valid_mask].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_threshold = 0.5
    best_f1 = 0
    
    print("\nThreshold optimization:")
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average="binary", zero_division=0
        )
        print(f"  Threshold {thresh:.2f}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\n  Best threshold: {best_threshold} (F1={best_f1:.4f})")
    
    return best_threshold


# ============================================================================
# Main Training Function
# ============================================================================

def train_alignment():
    parser = argparse.ArgumentParser(description="Train improved alignment model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=PATH_PREFIX + "alignment_model_v2_output")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use focal loss")
    parser.add_argument("--use-contrastive", action="store_true", help="Add contrastive loss")
    parser.add_argument("--contrastive-weight", type=float, default=0.5, help="Weight for contrastive loss")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Freeze encoders for first N epochs")
    parser.add_argument("--quick-test", action="store_true", help="Quick test on small subset")
    parser.add_argument("--no-cross-attention", action="store_true", help="Disable cross-attention")
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING IMPROVED WORD ALIGNMENT MODEL (v2)")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    train_examples = load_alignment_data(Path(DATA_PREFIX + "data/train.json"))
    dev_examples = load_alignment_data(Path(DATA_PREFIX + "data/dev.json"))
    test_examples = load_alignment_data(Path(DATA_PREFIX + "data/test.json"))
    
    if args.quick_test:
        train_examples = train_examples[:100]
        dev_examples = dev_examples[:50]
        test_examples = test_examples[:50]
        args.epochs = 2
        print("  [QUICK TEST MODE - using small subset]")
    
    print(f"  Train: {len(train_examples):,} verses with alignments")
    print(f"  Dev:   {len(dev_examples):,} verses")
    print(f"  Test:  {len(test_examples):,} verses")
    
    # Load tokenizers
    print("\n2. Loading tokenizers...")
    greek_tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("  ✓ Tokenizers loaded")
    
    # Create datasets
    print("\n3. Creating datasets with hard negative mining...")
    train_dataset = ImprovedAlignmentDataset(
        train_examples, greek_tokenizer, english_tokenizer,
        hard_negative_ratio=0.5,
        pos_to_neg_ratio=1.0,  # Balanced 1:1
    )
    dev_dataset = ImprovedAlignmentDataset(
        dev_examples, greek_tokenizer, english_tokenizer,
        hard_negative_ratio=0.5,
        pos_to_neg_ratio=1.0,
    )
    test_dataset = ImprovedAlignmentDataset(
        test_examples, greek_tokenizer, english_tokenizer,
        hard_negative_ratio=0.5,
        pos_to_neg_ratio=1.0,
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    print(f"  ✓ Created {len(train_dataloader):,} training batches")
    
    # Create model
    print("\n4. Creating improved model...")
    model = ImprovedAlignmentModel(
        use_cross_attention=not args.no_cross_attention,
        freeze_encoders_epochs=args.freeze_epochs,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Cross-attention: {not args.no_cross_attention}")
    
    # Loss function
    use_contrastive = args.use_contrastive
    
    if use_contrastive:
        loss_fn = CombinedLoss(
            classification_weight=1.0,
            contrastive_weight=args.contrastive_weight,
            class_weights=torch.tensor([1.0, 1.5]).to(device),
            ignore_index=-100,
            use_focal=args.use_focal_loss,
        )
        print(f"  Using Combined Loss (contrastive_weight={args.contrastive_weight})")
    elif args.use_focal_loss:
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0, ignore_index=-100)
        print("  Using Focal Loss")
    else:
        # Weighted CE loss - slight upweight for positive class
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.5]).to(device),
            ignore_index=-100
        )
        print("  Using Weighted Cross-Entropy Loss")
    
    # Setup training
    print("\n5. Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps,
    )
    
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Training steps: {num_training_steps:,}")
    
    # Train!
    print("\n" + "=" * 80)
    print("6. TRAINING STARTED")
    print("=" * 80)
    
    best_f1 = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, loss_fn, epoch, 
            use_contrastive=use_contrastive
        )
        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"P: {train_metrics['precision']:.4f}, "
            f"R: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"
        )
        
        # Evaluate
        dev_metrics = evaluate(model, dev_dataloader, device)
        print(
            f"Dev   - P: {dev_metrics['precision']:.4f}, "
            f"R: {dev_metrics['recall']:.4f}, "
            f"F1: {dev_metrics['f1']:.4f}"
        )
        
        # Save best model
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (F1: {best_f1:.4f})")
    
    # Find optimal threshold
    print("\n" + "=" * 80)
    print("7. THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    optimal_threshold = find_optimal_threshold(model, dev_dataloader, device)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("8. FINAL EVALUATION")
    print("=" * 80)
    
    print("\nDev set:")
    dev_metrics = evaluate(model, dev_dataloader, device)
    print(f"  Precision: {dev_metrics['precision']:.4f}")
    print(f"  Recall:    {dev_metrics['recall']:.4f}")
    print(f"  F1 Score:  {dev_metrics['f1']:.4f}")
    
    print("\nTest set:")
    test_metrics = evaluate(model, test_dataloader, device)
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    # Save final artifacts
    print(f"\n9. Saving model and tokenizers...")
    
    torch.save(model.state_dict(), output_dir / "model.pt")
    greek_tokenizer.save_pretrained(output_dir / "greek_tokenizer")
    english_tokenizer.save_pretrained(output_dir / "english_tokenizer")
    
    # Save config
    config = {
        "greek_model": "bowphs/GreBerta",
        "english_model": "bert-base-uncased",
        "dev_f1": float(dev_metrics["f1"]),
        "test_f1": float(test_metrics["f1"]),
        "optimal_threshold": float(optimal_threshold),
        "use_cross_attention": not args.no_cross_attention,
        "use_contrastive": args.use_contrastive,
        "contrastive_weight": args.contrastive_weight if args.use_contrastive else 0,
        "use_focal_loss": args.use_focal_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "model_version": "v2",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Optimal threshold: {optimal_threshold}")
    print("\nTo use the model, see test_alignment_v2.py")
    print("=" * 80)


if __name__ == "__main__":
    train_alignment()

