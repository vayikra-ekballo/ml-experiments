#!/usr/bin/env python3
"""
Fine-tune GreBerta for POS Tagging on Koine Greek (NT)

This script fine-tunes GreBerta on part-of-speech tagging using the NT dataset.
This is a simpler task than alignment and serves as a good starting point.

Usage:
    python train_pos_tagger.py [--epochs 3] [--batch-size 16] [--lr 2e-5]
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List

print("✓ Standard library imports complete")

import torch
print("✓ PyTorch imported")

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
print("✓ Transformers imported")

from datasets import Dataset, DatasetDict
print("✓ Datasets imported")

import numpy as np
print("✓ NumPy imported")

from sklearn.metrics import classification_report, accuracy_score
print("✓ Sklearn imported")

print("=" * 80)
print("FINE-TUNING GREBERTA FOR POS TAGGING")
print("=" * 80)

def load_pos_data(filepath: Path) -> List[Dict]:
    """Load and extract POS tagging data from alignment JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pos_data = []
    for verse in data['data']:
        tokens = [t['word'] for t in verse['greek_tokens']]
        pos_tags = [t['pos'] for t in verse['greek_tokens']]
        
        if tokens:  # Skip empty verses
            pos_data.append({
                'tokens': tokens,
                'pos_tags': pos_tags,
                'verse_id': verse['verse_id']
            })
    
    return pos_data


def create_label_mapping(train_data: List[Dict]) -> Dict[str, int]:
    """Create mapping from POS tags to integer IDs"""
    all_tags = set()
    for example in train_data:
        all_tags.update(example['pos_tags'])
    
    # Sort for consistency
    sorted_tags = sorted(all_tags)
    tag_to_id = {tag: i for i, tag in enumerate(sorted_tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}
    
    return tag_to_id, id_to_tag


def tokenize_and_align_labels(examples, tokenizer, tag_to_id):
    """
    Tokenize text and align POS labels with subword tokens.
    
    When a word is split into subwords (e.g., 'λόγος' -> ['λ', '##όγος']),
    we assign the label to the first subword and -100 to the rest (ignored in loss).
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=512
    )
    
    labels = []
    for i, label_list in enumerate(examples['pos_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            # Special tokens get -100 (ignored in loss)
            if word_idx is None:
                label_ids.append(-100)
            # First subword of each word gets the label
            elif word_idx != previous_word_idx:
                label_ids.append(tag_to_id[label_list[word_idx]])
            # Other subwords get -100
            else:
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics(eval_pred, id_to_tag):
    """Compute accuracy and per-class metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten for metrics
    flat_predictions = [tag for sent in true_predictions for tag in sent]
    flat_labels = [tag for sent in true_labels for tag in sent]
    
    accuracy = accuracy_score(flat_labels, flat_predictions)
    
    return {
        'accuracy': accuracy,
        'num_examples': len(flat_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GreBerta for POS tagging')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--model-name', default='bowphs/GreBerta', help='Base model name')
    parser.add_argument('--output-dir', default='./pos_tagger_output', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 80)
    print("FINE-TUNING GREBERTA FOR POS TAGGING")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading data...")
    train_data = load_pos_data(Path('data/train.json'))
    dev_data = load_pos_data(Path('data/dev.json'))
    test_data = load_pos_data(Path('data/test.json'))
    
    print(f"  Train: {len(train_data):,} verses")
    print(f"  Dev:   {len(dev_data):,} verses")
    print(f"  Test:  {len(test_data):,} verses")
    
    # 2. Create label mapping
    print("\n2. Creating label mapping...")
    tag_to_id, id_to_tag = create_label_mapping(train_data)
    num_labels = len(tag_to_id)
    print(f"  Found {num_labels} POS tags:")
    for tag, idx in sorted(tag_to_id.items(), key=lambda x: x[1]):
        # Count occurrences in train
        count = sum(1 for ex in train_data for t in ex['pos_tags'] if t == tag)
        print(f"    {idx:2d}. {tag:5s} ({count:,} tokens)")
    
    # 3. Load tokenizer and model
    print(f"\n3. Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id_to_tag,
        label2id=tag_to_id
    )
    print(f"  ✓ Model loaded with {num_labels} labels")
    print(f"  ✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 4. Create datasets
    print("\n4. Creating Hugging Face datasets...")
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Tokenize
    print("  Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag_to_id),
        batched=True,
        remove_columns=['tokens', 'pos_tags', 'verse_id']
    )
    dev_dataset = dev_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag_to_id),
        batched=True,
        remove_columns=['tokens', 'pos_tags', 'verse_id']
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag_to_id),
        batched=True,
        remove_columns=['tokens', 'pos_tags', 'verse_id']
    )
    print(f"  ✓ Tokenized {len(train_dataset):,} training examples")
    
    # 5. Setup training
    print(f"\n5. Setting up training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output dir: {args.output_dir}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=50,
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id_to_tag),
    )
    
    # 6. Train!
    print("\n" + "=" * 80)
    print("6. TRAINING STARTED")
    print("=" * 80)
    
    train_result = trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value}")
    
    # 7. Evaluate on dev set
    print("\n7. Evaluating on dev set...")
    dev_results = trainer.evaluate(eval_dataset=dev_dataset)
    print(f"Dev accuracy: {dev_results['eval_accuracy']:.4f}")
    
    # 8. Evaluate on test set
    print("\n8. Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    
    # 9. Save model
    print(f"\n9. Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save label mappings
    import json
    with open(Path(args.output_dir) / 'label_mapping.json', 'w') as f:
        json.dump({'tag_to_id': tag_to_id, 'id_to_tag': id_to_tag}, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Dev accuracy:   {dev_results['eval_accuracy']:.4f}")
    print(f"Test accuracy:  {test_results['eval_accuracy']:.4f}")
    print("\nTo use the model:")
    print(f"  from transformers import AutoModelForTokenClassification, AutoTokenizer")
    print(f"  model = AutoModelForTokenClassification.from_pretrained('{args.output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
    print("=" * 80)


if __name__ == '__main__':
    main()

