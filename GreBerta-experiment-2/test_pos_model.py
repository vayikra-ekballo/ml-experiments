#!/usr/bin/env python3
"""
Test the fine-tuned POS tagger on sample Greek text

Usage:
    python test_pos_model.py
    python test_pos_model.py --text "Ἐν ἀρχῇ ἦν ὁ λόγος"
    python test_pos_model.py --model-dir ./pos_tagger_output
"""

import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch


def predict_pos_tags(text: str, model, tokenizer):
    """Predict POS tags for Greek text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.tokenize(text)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1)[0]
    
    # Map predictions to labels
    results = []
    token_idx = 0
    for word_id in inputs.word_ids():
        if word_id is not None and (not results or word_id != results[-1]['word_id']):
            # First subword of each word
            label_id = predictions[token_idx].item()
            label = model.config.id2label[label_id]
            results.append({
                'word_id': word_id,
                'token': tokens[token_idx] if token_idx < len(tokens) else '',
                'label': label
            })
        token_idx += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test POS tagger')
    parser.add_argument('--model-dir', default='./pos_tagger_output', help='Model directory')
    parser.add_argument('--text', default=None, help='Greek text to tag')
    args = parser.parse_args()
    
    print("=" * 80)
    print("POS TAGGER INFERENCE")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nHave you trained the model yet?")
        print("Run: python train_pos_tagger.py")
        return
    
    # Test examples
    if args.text:
        test_examples = [args.text]
    else:
        test_examples = [
            "Ἐν ἀρχῇ ἦν ὁ λόγος",  # John 1:1
            "καὶ ὁ λόγος ἦν πρὸς τὸν θεόν",  # John 1:1 continued
            "πάντα δι' αὐτοῦ ἐγένετο",  # John 1:3
            "ἐγὼ εἰμι ἡ ὁδὸς καὶ ἡ ἀλήθεια καὶ ἡ ζωή",  # John 14:6
        ]
    
    # POS tag meanings
    pos_meanings = {
        'V-': 'Verb',
        'N-': 'Noun',
        'RA': 'Article',
        'C-': 'Conjunction',
        'RP': 'Personal Pronoun',
        'P-': 'Preposition',
        'A-': 'Adjective',
        'D-': 'Adverb',
        'RD': 'Demonstrative Pronoun',
        'RR': 'Relative Pronoun',
        'RI': 'Interrogative Pronoun',
        'X-': 'Particle',
        'I-': 'Interjection',
    }
    
    print("\nTesting on sample verses:")
    print("=" * 80)
    
    for text in test_examples:
        print(f"\nText: {text}")
        print("-" * 80)
        
        results = predict_pos_tags(text, model, tokenizer)
        
        # Reconstruct words and their tags
        words = text.split()
        for i, word in enumerate(words):
            matching_result = [r for r in results if r['word_id'] == i]
            if matching_result:
                tag = matching_result[0]['label']
                meaning = pos_meanings.get(tag, 'Unknown')
                print(f"  {word:20s} -> {tag:5s} ({meaning})")
    
    print("\n" + "=" * 80)
    print("POS Tag Legend:")
    print("=" * 80)
    for tag, meaning in sorted(pos_meanings.items()):
        print(f"  {tag:5s} = {meaning}")
    
    print("\n" + "=" * 80)
    print("✓ Testing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

