#!/usr/bin/env python3
"""
Fine-tune GreBerta + BERT for Greek-English Word Alignment

This script trains a cross-lingual alignment model using:
- GreBerta for encoding Greek text
- BERT for encoding English text
- Token-pair classification to predict alignments

Usage:
    python train_alignment.py [--epochs 3] [--batch-size 8] [--lr 2e-5]
    python train_alignment.py --quick-test  # Train on subset for testing
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report


try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    path_prefix = "/content/drive/MyDrive/UofT DL Term project/GreBerta-experiment-2/"
else:
    path_prefix = ""


@dataclass
class AlignmentExample:
    """Single verse with alignment pairs"""

    verse_id: str
    greek_tokens: List[str]
    english_tokens: List[str]
    alignments: List[Tuple[int, int]]  # (greek_idx, english_idx) pairs


class AlignmentModel(nn.Module):
    """Cross-lingual word alignment model"""

    def __init__(
        self,
        greek_model_name="bowphs/GreBerta",
        english_model_name="bert-base-uncased",
        hidden_dim=256,
    ):
        super().__init__()

        # Encoders
        self.greek_encoder = AutoModel.from_pretrained(greek_model_name)
        self.english_encoder = AutoModel.from_pretrained(english_model_name)

        # Get hidden sizes
        greek_hidden = self.greek_encoder.config.hidden_size
        english_hidden = self.english_encoder.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(greek_hidden + english_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),  # Binary: aligned or not
        )

    def forward(
        self,
        greek_input_ids,
        greek_attention_mask,
        english_input_ids,
        english_attention_mask,
        greek_indices,
        english_indices,
    ):
        """
        Args:
            greek_input_ids: [batch_size, max_greek_len]
            greek_attention_mask: [batch_size, max_greek_len]
            english_input_ids: [batch_size, max_english_len]
            english_attention_mask: [batch_size, max_english_len]
            greek_indices: [batch_size, num_pairs] - which Greek token for each pair
            english_indices: [batch_size, num_pairs] - which English token for each pair

        Returns:
            logits: [batch_size, num_pairs, 2] - alignment scores
        """
        # Encode Greek
        greek_outputs = self.greek_encoder(
            input_ids=greek_input_ids, attention_mask=greek_attention_mask
        )
        greek_embeddings = greek_outputs.last_hidden_state  # [batch, greek_len, hidden]

        # Encode English
        english_outputs = self.english_encoder(
            input_ids=english_input_ids, attention_mask=english_attention_mask
        )
        english_embeddings = (
            english_outputs.last_hidden_state
        )  # [batch, eng_len, hidden]

        # Gather embeddings for specified pairs
        batch_size, num_pairs = greek_indices.shape

        # Get Greek embeddings for each pair
        greek_pair_embeddings = torch.gather(
            greek_embeddings,
            dim=1,
            index=greek_indices.unsqueeze(-1).expand(-1, -1, greek_embeddings.size(-1)),
        )  # [batch, num_pairs, greek_hidden]

        # Get English embeddings for each pair
        english_pair_embeddings = torch.gather(
            english_embeddings,
            dim=1,
            index=english_indices.unsqueeze(-1).expand(
                -1, -1, english_embeddings.size(-1)
            ),
        )  # [batch, num_pairs, english_hidden]

        # Concatenate embeddings
        combined = torch.cat([greek_pair_embeddings, english_pair_embeddings], dim=-1)

        # Classify each pair
        logits = self.classifier(combined)  # [batch, num_pairs, 2]

        return logits


class AlignmentDataset(Dataset):
    """Dataset for word alignment training"""

    def __init__(
        self,
        examples: List[AlignmentExample],
        greek_tokenizer,
        english_tokenizer,
        max_pairs_per_verse=50,
    ):
        self.examples = examples
        self.greek_tokenizer = greek_tokenizer
        self.english_tokenizer = english_tokenizer
        self.max_pairs_per_verse = max_pairs_per_verse

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize Greek (join with spaces)
        greek_text = " ".join(example.greek_tokens)
        greek_encoded = self.greek_tokenizer(
            greek_text, truncation=True, max_length=512, return_tensors="pt"
        )

        # Tokenize English
        english_text = " ".join(example.english_tokens)
        english_encoded = self.english_tokenizer(
            english_text, truncation=True, max_length=512, return_tensors="pt"
        )

        # Map original word indices to subword indices
        greek_word_to_token = self._get_word_to_token_map(
            example.greek_tokens, greek_encoded
        )
        english_word_to_token = self._get_word_to_token_map(
            example.english_tokens, english_encoded
        )

        # Create training pairs
        # Positive examples: actual alignments
        positive_pairs = []
        for greek_idx, english_idx in example.alignments:
            if (
                greek_idx in greek_word_to_token
                and english_idx in english_word_to_token
            ):
                positive_pairs.append(
                    (
                        greek_word_to_token[greek_idx],
                        english_word_to_token[english_idx],
                        1,  # label: aligned
                    )
                )

        # Negative examples: random non-aligned pairs
        num_negatives = min(len(positive_pairs) * 2, self.max_pairs_per_verse)
        negative_pairs = []

        all_greek_indices = list(greek_word_to_token.values())
        all_english_indices = list(english_word_to_token.values())

        # Create set of positive pairs for fast lookup
        positive_set = {(g, e) for g, e, _ in positive_pairs}

        attempts = 0
        while len(negative_pairs) < num_negatives and attempts < num_negatives * 10:
            g_idx = random.choice(all_greek_indices)
            e_idx = random.choice(all_english_indices)
            if (g_idx, e_idx) not in positive_set:
                negative_pairs.append((g_idx, e_idx, 0))  # label: not aligned
            attempts += 1

        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        # Limit total pairs
        all_pairs = all_pairs[: self.max_pairs_per_verse]

        if not all_pairs:
            # Create dummy pair if no valid pairs
            all_pairs = [(1, 1, 0)]

        greek_indices = torch.tensor([p[0] for p in all_pairs], dtype=torch.long)
        english_indices = torch.tensor([p[1] for p in all_pairs], dtype=torch.long)
        labels = torch.tensor([p[2] for p in all_pairs], dtype=torch.long)

        return {
            "greek_input_ids": greek_encoded["input_ids"].squeeze(0),
            "greek_attention_mask": greek_encoded["attention_mask"].squeeze(0),
            "english_input_ids": english_encoded["input_ids"].squeeze(0),
            "english_attention_mask": english_encoded["attention_mask"].squeeze(0),
            "greek_indices": greek_indices,
            "english_indices": english_indices,
            "labels": labels,
            "verse_id": example.verse_id,
        }

    def _get_word_to_token_map(self, words, encoded):
        """Map word indices to their first subword token index"""
        word_to_token = {}
        word_ids = encoded.word_ids()

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_to_token:
                word_to_token[word_idx] = token_idx

        return word_to_token


def collate_fn(batch):
    """Custom collate function for batching"""
    # Find max lengths
    max_greek_len = max(item["greek_input_ids"].size(0) for item in batch)
    max_english_len = max(item["english_input_ids"].size(0) for item in batch)
    max_pairs = max(item["labels"].size(0) for item in batch)

    # Pad everything
    greek_input_ids = []
    greek_attention_mask = []
    english_input_ids = []
    english_attention_mask = []
    greek_indices = []
    english_indices = []
    labels = []
    verse_ids = []

    for item in batch:
        # Pad Greek
        g_len = item["greek_input_ids"].size(0)
        greek_input_ids.append(
            torch.cat(
                [
                    item["greek_input_ids"],
                    torch.zeros(max_greek_len - g_len, dtype=torch.long),
                ]
            )
        )
        greek_attention_mask.append(
            torch.cat(
                [
                    item["greek_attention_mask"],
                    torch.zeros(max_greek_len - g_len, dtype=torch.long),
                ]
            )
        )

        # Pad English
        e_len = item["english_input_ids"].size(0)
        english_input_ids.append(
            torch.cat(
                [
                    item["english_input_ids"],
                    torch.zeros(max_english_len - e_len, dtype=torch.long),
                ]
            )
        )
        english_attention_mask.append(
            torch.cat(
                [
                    item["english_attention_mask"],
                    torch.zeros(max_english_len - e_len, dtype=torch.long),
                ]
            )
        )

        # Pad pairs
        num_pairs = item["labels"].size(0)
        greek_indices.append(
            torch.cat(
                [
                    item["greek_indices"],
                    torch.zeros(max_pairs - num_pairs, dtype=torch.long),
                ]
            )
        )
        english_indices.append(
            torch.cat(
                [
                    item["english_indices"],
                    torch.zeros(max_pairs - num_pairs, dtype=torch.long),
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    item["labels"],
                    torch.full((max_pairs - num_pairs,), -100, dtype=torch.long),
                ]
            )  # -100 = ignore
        )

        verse_ids.append(item["verse_id"])

    return {
        "greek_input_ids": torch.stack(greek_input_ids),
        "greek_attention_mask": torch.stack(greek_attention_mask),
        "english_input_ids": torch.stack(english_input_ids),
        "english_attention_mask": torch.stack(english_attention_mask),
        "greek_indices": torch.stack(greek_indices),
        "english_indices": torch.stack(english_indices),
        "labels": torch.stack(labels),
        "verse_ids": verse_ids,
    }


def load_alignment_data(filepath: Path) -> List[AlignmentExample]:
    """Load alignment data from JSON"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for verse in data["data"]:
        if not verse["alignments"]:
            continue

        greek_tokens = [t["word"] for t in verse["greek_tokens"]]
        english_tokens = [t["word"] for t in verse["english_tokens"]]
        alignments = [(a["greek_idx"], a["english_idx"]) for a in verse["alignments"]]

        examples.append(
            AlignmentExample(
                verse_id=verse["verse_id"],
                greek_tokens=greek_tokens,
                english_tokens=english_tokens,
                alignments=alignments,
            )
        )

    return examples


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move to device
        greek_input_ids = batch["greek_input_ids"].to(device)
        greek_attention_mask = batch["greek_attention_mask"].to(device)
        english_input_ids = batch["english_input_ids"].to(device)
        english_attention_mask = batch["english_attention_mask"].to(device)
        greek_indices = batch["greek_indices"].to(device)
        english_indices = batch["english_indices"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(
            greek_input_ids,
            greek_attention_mask,
            english_input_ids,
            english_attention_mask,
            greek_indices,
            english_indices,
        )

        # Compute loss (only on valid pairs)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
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
        all_labels, all_preds, average="binary"
    )

    return {
        "loss": total_loss / len(dataloader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            greek_input_ids = batch["greek_input_ids"].to(device)
            greek_attention_mask = batch["greek_attention_mask"].to(device)
            english_input_ids = batch["english_input_ids"].to(device)
            english_attention_mask = batch["english_attention_mask"].to(device)
            greek_indices = batch["greek_indices"].to(device)
            english_indices = batch["english_indices"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                greek_input_ids,
                greek_attention_mask,
                english_input_ids,
                english_attention_mask,
                greek_indices,
                english_indices,
            )

            preds = torch.argmax(logits, dim=-1)
            valid_mask = labels != -100
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def train_alignment():
    args = {
        "epochs": 3,
        "batch_size": 8,
        "lr": 2e-5,
        "output_dir": path_prefix + "alignment_model_output_colab",
    }

    print("=" * 80)
    print("TRAINING WORD ALIGNMENT MODEL")
    print("=" * 80)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\n1. Loading data...")
    train_examples = load_alignment_data(Path(path_prefix + "data/train.json"))
    dev_examples = load_alignment_data(Path(path_prefix + "data/dev.json"))
    test_examples = load_alignment_data(Path(path_prefix + "data/test.json"))

    print(f"  Train: {len(train_examples):,} verses with alignments")
    print(f"  Dev:   {len(dev_examples):,} verses")
    print(f"  Test:  {len(test_examples):,} verses")

    # Load tokenizers
    print("\n2. Loading tokenizers...")
    greek_tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("  ✓ Tokenizers loaded")

    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = AlignmentDataset(train_examples, greek_tokenizer, english_tokenizer)
    dev_dataset = AlignmentDataset(dev_examples, greek_tokenizer, english_tokenizer)
    test_dataset = AlignmentDataset(test_examples, greek_tokenizer, english_tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    print(f"  ✓ Created {len(train_dataloader):,} training batches")

    # Create model
    print("\n4. Creating model...")
    model = AlignmentModel()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup training
    print("\n5. Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])
    num_training_steps = len(train_dataloader) * args["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps,
    )

    print(f"  Epochs: {args['epochs']}")
    print(f"  Batch size: {args['batch_size']}")
    print(f"  Learning rate: {args['lr']}")
    print(f"  Training steps: {num_training_steps:,}")

    # Train!
    print("\n" + "=" * 80)
    print("6. TRAINING STARTED")
    print("=" * 80)

    best_f1 = 0
    for epoch in range(args["epochs"]):
        print(f"\nEpoch {epoch + 1}/{args['epochs']}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
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
            output_dir = Path(args["output_dir"])
            output_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (F1: {best_f1:.4f})")

    # Final evaluation
    print("\n" + "=" * 80)
    print("7. FINAL EVALUATION")
    print("=" * 80)

    # Load best model
    model.load_state_dict(torch.load(Path(args["output_dir"]) / "best_model.pt"))

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
    print(f"\n8. Saving model and tokenizers...")
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")
    greek_tokenizer.save_pretrained(output_dir / "greek_tokenizer")
    english_tokenizer.save_pretrained(output_dir / "english_tokenizer")

    # Save config
    config = {
        "greek_model": "bowphs/GreBerta",
        "english_model": "bert-base-uncased",
        "dev_f1": dev_metrics["f1"],
        "test_f1": test_metrics["f1"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {args['output_dir']}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print("\nTo use the model, see test_alignment.py")
    print("=" * 80)


# train_alignment()
