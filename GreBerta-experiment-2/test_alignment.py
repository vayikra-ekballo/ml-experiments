#!/usr/bin/env python3
"""
Test the trained word alignment model and generate interlinear translations

Usage:
    python test_alignment.py --greek "Ἐν ἀρχῇ ἦν ὁ λόγος" --english "In the beginning was the Word"
    python test_alignment.py --verse "John 1:1"
    python test_alignment.py --interactive
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

# Import model from training script
from train_alignment import AlignmentModel


class AlignmentPredictor:
    """Wrapper for trained alignment model"""
    
    def __init__(self, model_dir='./alignment_model_output'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(Path(model_dir) / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Load tokenizers
        self.greek_tokenizer = AutoTokenizer.from_pretrained(
            Path(model_dir) / 'greek_tokenizer'
        )
        self.english_tokenizer = AutoTokenizer.from_pretrained(
            Path(model_dir) / 'english_tokenizer'
        )
        
        # Load model
        self.model = AlignmentModel(
            greek_model_name=config['greek_model'],
            english_model_name=config['english_model']
        )
        self.model.load_state_dict(
            torch.load(Path(model_dir) / 'model.pt', map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_dir}")
        print(f"✓ Test F1 Score: {config['test_f1']:.4f}")
    
    def align(self, greek_text: str, english_text: str, 
              threshold=0.5) -> List[Tuple[int, int, float]]:
        """
        Predict alignments between Greek and English text
        
        Returns:
            List of (greek_word_idx, english_word_idx, confidence_score)
        """
        # Split into words
        greek_words = greek_text.split()
        english_words = english_text.split()
        
        # Tokenize
        greek_encoded = self.greek_tokenizer(
            greek_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        english_encoded = self.english_tokenizer(
            english_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Map words to tokens
        greek_word_to_token = self._get_word_to_token_map(
            greek_words, greek_encoded
        )
        english_word_to_token = self._get_word_to_token_map(
            english_words, english_encoded
        )
        
        # Create all possible pairs
        pairs = []
        greek_indices_list = []
        english_indices_list = []
        
        for g_word_idx, g_token_idx in greek_word_to_token.items():
            for e_word_idx, e_token_idx in english_word_to_token.items():
                pairs.append((g_word_idx, e_word_idx))
                greek_indices_list.append(g_token_idx)
                english_indices_list.append(e_token_idx)
        
        if not pairs:
            return []
        
        # Batch predict all pairs
        greek_indices = torch.tensor([greek_indices_list], dtype=torch.long).to(self.device)
        english_indices = torch.tensor([english_indices_list], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(
                greek_encoded['input_ids'],
                greek_encoded['attention_mask'],
                english_encoded['input_ids'],
                english_encoded['attention_mask'],
                greek_indices,
                english_indices
            )
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            aligned_probs = probs[0, :, 1].cpu().numpy()  # Probability of being aligned
        
        # Filter by threshold
        alignments = []
        for (g_idx, e_idx), prob in zip(pairs, aligned_probs):
            if prob >= threshold:
                alignments.append((g_idx, e_idx, float(prob)))
        
        # Sort by confidence
        alignments.sort(key=lambda x: x[2], reverse=True)
        
        return alignments
    
    def _get_word_to_token_map(self, words, encoded):
        """Map word indices to their first subword token index"""
        word_to_token = {}
        word_ids = encoded.word_ids()
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_to_token:
                word_to_token[word_idx] = token_idx
        
        return word_to_token


def generate_interlinear(greek_text: str, english_text: str, 
                         alignments: List[Tuple[int, int, float]],
                         show_confidence=False):
    """Generate interlinear display"""
    greek_words = greek_text.split()
    english_words = english_text.split()
    
    # Create alignment dict
    greek_to_english = {}
    for g_idx, e_idx, conf in alignments:
        if g_idx not in greek_to_english:
            greek_to_english[g_idx] = []
        greek_to_english[g_idx].append((english_words[e_idx], conf))
    
    print("\n" + "=" * 80)
    print("INTERLINEAR DISPLAY")
    print("=" * 80)
    
    for i, greek_word in enumerate(greek_words):
        aligned_english = greek_to_english.get(i, [])
        
        if aligned_english:
            english_str = ", ".join(
                f"{word}" + (f" ({conf:.2f})" if show_confidence else "")
                for word, conf in aligned_english
            )
        else:
            english_str = "[no alignment]"
        
        print(f"{i+1:2d}. {greek_word:20s} → {english_str}")
    
    print("=" * 80)


def load_verse_from_data(verse_ref: str):
    """Load a verse from the test data"""
    # Parse reference like "John 1:1"
    parts = verse_ref.rsplit(' ', 1)
    if len(parts) != 2:
        return None, None
    
    book_name = parts[0]
    try:
        chapter, verse = map(int, parts[1].split(':'))
    except:
        return None, None
    
    # Search in test data
    test_file = Path('data/test.json')
    if not test_file.exists():
        return None, None
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    for verse_data in data['data']:
        if (verse_data['book_name'] == book_name and
            verse_data['chapter'] == chapter and
            verse_data['verse'] == verse):
            
            greek_text = ' '.join(t['word'] for t in verse_data['greek_tokens'])
            english_text = ' '.join(t['word'] for t in verse_data['english_tokens'])
            return greek_text, english_text
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Test alignment model')
    parser.add_argument('--model-dir', default='./alignment_model_output')
    parser.add_argument('--greek', help='Greek text to align')
    parser.add_argument('--english', help='English text to align')
    parser.add_argument('--verse', help='Verse reference (e.g., "John 1:1")')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for alignments')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Show confidence scores')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    args = parser.parse_args()
    
    print("=" * 80)
    print("WORD ALIGNMENT INFERENCE")
    print("=" * 80)
    
    # Load model
    try:
        predictor = AlignmentPredictor(args.model_dir)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nHave you trained the model yet?")
        print("Run: python train_alignment.py")
        return
    
    # Interactive mode
    if args.interactive:
        print("\nInteractive Mode - Enter Greek and English text to align")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                greek = input("Greek text: ").strip()
                if greek.lower() == 'quit':
                    break
                
                english = input("English text: ").strip()
                if english.lower() == 'quit':
                    break
                
                if greek and english:
                    alignments = predictor.align(greek, english, args.threshold)
                    generate_interlinear(greek, english, alignments, args.show_confidence)
                    print(f"\nFound {len(alignments)} alignments\n")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
        return
    
    # Get text
    if args.verse:
        greek, english = load_verse_from_data(args.verse)
        if greek is None:
            print(f"\n✗ Verse not found: {args.verse}")
            print("\nTry one of these:")
            print("  - 'Jude 1'")
            print("  - '3 John 1'")
            print("  - 'Revelation 1:1'")
            return
        print(f"\nVerse: {args.verse}")
    elif args.greek and args.english:
        greek = args.greek
        english = args.english
    else:
        # Default examples
        examples = [
            ("Ἐν ἀρχῇ ἦν ὁ λόγος", "In the beginning was the Word"),
            ("καὶ ὁ λόγος ἦν πρὸς τὸν θεόν", "and the Word was with God"),
            ("πάντα δι' αὐτοῦ ἐγένετο", "All things were made through him"),
        ]
        
        print("\nTesting on example verses...")
        for greek, english in examples:
            print(f"\nGreek:   {greek}")
            print(f"English: {english}")
            alignments = predictor.align(greek, english, args.threshold)
            generate_interlinear(greek, english, alignments, args.show_confidence)
        return
    
    # Predict alignments
    print(f"\nGreek:   {greek}")
    print(f"English: {english}")
    print(f"\nPredicting alignments (threshold: {args.threshold})...")
    
    alignments = predictor.align(greek, english, args.threshold)
    
    # Display
    generate_interlinear(greek, english, alignments, args.show_confidence)
    
    print(f"\n✓ Found {len(alignments)} alignments")
    
    # Show alignment list
    if alignments:
        print("\nAlignment pairs:")
        greek_words = greek.split()
        english_words = english.split()
        for g_idx, e_idx, conf in alignments[:10]:
            print(f"  {greek_words[g_idx]:15s} → {english_words[e_idx]:15s} (confidence: {conf:.3f})")
        if len(alignments) > 10:
            print(f"  ... and {len(alignments) - 10} more")


if __name__ == '__main__':
    main()

