#!/usr/bin/env python3
"""
Test the improved word alignment model (v2) and generate interlinear translations

Key improvements:
- Uses improved model architecture with cross-attention
- Applies optimal threshold from training
- Post-processing constraints for better alignment quality
- Multiple alignment selection strategies

Usage:
    python test_alignment_v2.py --greek "Ἐν ἀρχῇ ἦν ὁ λόγος" --english "In the beginning was the Word"
    python test_alignment_v2.py --verse "John 1:1"
    python test_alignment_v2.py --interactive
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import model from training script
from train_alignment_v2 import ImprovedAlignmentModel


class ImprovedAlignmentPredictor:
    """Wrapper for trained improved alignment model with post-processing"""
    
    def __init__(self, model_dir='./alignment_model_v2_output'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        
        # Load config
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Get optimal threshold (default to 0.5 if not found)
        self.threshold = self.config.get('optimal_threshold', 0.5)
        
        # Load tokenizers
        self.greek_tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / 'greek_tokenizer'
        )
        self.english_tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / 'english_tokenizer'
        )
        
        # Load model
        self.model = ImprovedAlignmentModel(
            greek_model_name=self.config['greek_model'],
            english_model_name=self.config['english_model'],
            use_cross_attention=self.config.get('use_cross_attention', True),
        )
        self.model.load_state_dict(
            torch.load(self.model_dir / 'model.pt', map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_dir}")
        print(f"✓ Model version: {self.config.get('model_version', 'v1')}")
        print(f"✓ Test F1 Score: {self.config.get('test_f1', 'N/A')}")
        print(f"✓ Optimal threshold: {self.threshold}")
    
    def align(
        self, 
        greek_text: str, 
        english_text: str,
        threshold: Optional[float] = None,
        strategy: str = 'greedy',  # 'greedy', 'best_match', 'all'
        ensure_coverage: bool = True,  # Ensure each content word has alignment
    ) -> List[Tuple[int, int, float]]:
        """
        Predict alignments between Greek and English text
        
        Args:
            greek_text: Greek text (space-separated words)
            english_text: English text (space-separated words)
            threshold: Confidence threshold (uses optimal if None)
            strategy: Alignment selection strategy
                - 'greedy': Take all above threshold
                - 'best_match': For each Greek word, take best English match
                - 'all': Return all pairs with their probabilities
            ensure_coverage: Post-process to ensure content words have alignments
        
        Returns:
            List of (greek_word_idx, english_word_idx, confidence_score)
        """
        if threshold is None:
            threshold = self.threshold
        
        # Split into words
        greek_words = greek_text.split()
        english_words = english_text.split()
        
        num_greek = len(greek_words)
        num_english = len(english_words)
        
        if num_greek == 0 or num_english == 0:
            return []
        
        # Tokenize
        greek_encoded = self.greek_tokenizer(
            greek_text, return_tensors='pt', truncation=True, max_length=512
        ).to(self.device)
        
        english_encoded = self.english_tokenizer(
            english_text, return_tensors='pt', truncation=True, max_length=512
        ).to(self.device)
        
        # Get word spans
        greek_word_spans = self._get_word_spans(greek_words, greek_encoded)
        english_word_spans = self._get_word_spans(english_words, english_encoded)
        
        # Create all possible pairs
        all_pairs = []
        for g_idx in range(num_greek):
            for e_idx in range(num_english):
                all_pairs.append((g_idx, e_idx))
        
        if not all_pairs:
            return []
        
        # Prepare tensors
        pair_indices = torch.tensor([all_pairs], dtype=torch.long).to(self.device)
        greek_word_spans_t = torch.tensor([greek_word_spans], dtype=torch.long).to(self.device)
        english_word_spans_t = torch.tensor([english_word_spans], dtype=torch.long).to(self.device)
        greek_positions = torch.tensor([[p[0] for p in all_pairs]], dtype=torch.long).to(self.device)
        english_positions = torch.tensor([[p[1] for p in all_pairs]], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(
                greek_encoded['input_ids'],
                greek_encoded['attention_mask'],
                english_encoded['input_ids'],
                english_encoded['attention_mask'],
                greek_word_spans_t,
                english_word_spans_t,
                pair_indices,
                greek_positions,
                english_positions,
            )
            
            probs = F.softmax(logits, dim=-1)
            aligned_probs = probs[0, :, 1].cpu().numpy()  # Probability of being aligned
        
        # Create probability matrix
        prob_matrix = {}
        for (g_idx, e_idx), prob in zip(all_pairs, aligned_probs):
            prob_matrix[(g_idx, e_idx)] = float(prob)
        
        # Apply selection strategy
        if strategy == 'all':
            alignments = [(g, e, prob_matrix[(g, e)]) for g, e in all_pairs]
        elif strategy == 'best_match':
            alignments = self._best_match_strategy(
                num_greek, num_english, prob_matrix, threshold
            )
        else:  # greedy
            alignments = self._greedy_strategy(
                num_greek, num_english, prob_matrix, threshold
            )
        
        # Post-processing: ensure coverage for content words
        if ensure_coverage and strategy != 'all':
            alignments = self._ensure_coverage(
                alignments, greek_words, english_words, prob_matrix, threshold
            )
        
        # Sort by confidence
        alignments.sort(key=lambda x: x[2], reverse=True)
        
        return alignments
    
    def _get_word_spans(self, words, encoded):
        """Get (start, end) token indices for each word"""
        word_ids = encoded.word_ids()
        spans = []
        
        for word_idx in range(len(words)):
            token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            if token_indices:
                spans.append((min(token_indices), max(token_indices) + 1))
            else:
                spans.append((0, 1))  # Fallback to first token
        
        return spans
    
    def _greedy_strategy(self, num_greek, num_english, prob_matrix, threshold):
        """Take all pairs above threshold"""
        alignments = []
        for g in range(num_greek):
            for e in range(num_english):
                prob = prob_matrix.get((g, e), 0)
                if prob >= threshold:
                    alignments.append((g, e, prob))
        return alignments
    
    def _best_match_strategy(self, num_greek, num_english, prob_matrix, threshold):
        """For each Greek word, take best English match if above threshold"""
        alignments = []
        
        for g in range(num_greek):
            best_e = -1
            best_prob = threshold
            
            for e in range(num_english):
                prob = prob_matrix.get((g, e), 0)
                if prob > best_prob:
                    best_prob = prob
                    best_e = e
            
            if best_e >= 0:
                alignments.append((g, best_e, best_prob))
        
        return alignments
    
    def _ensure_coverage(self, alignments, greek_words, english_words, prob_matrix, threshold):
        """
        Post-processing: ensure content words have at least one alignment
        by lowering threshold for unaligned words
        """
        # Find aligned Greek words
        aligned_greek = set(a[0] for a in alignments)
        
        # Content word detection (simple heuristic)
        greek_content_indices = [
            i for i, word in enumerate(greek_words)
            if len(word) > 2 and not self._is_greek_function_word(word)
        ]
        
        # Find unaligned content words
        unaligned_content = [i for i in greek_content_indices if i not in aligned_greek]
        
        # For each unaligned content word, find best match with lower threshold
        additional = []
        for g in unaligned_content:
            best_e = -1
            best_prob = 0.1  # Lower threshold for coverage
            
            for e in range(len(english_words)):
                prob = prob_matrix.get((g, e), 0)
                if prob > best_prob:
                    best_prob = prob
                    best_e = e
            
            if best_e >= 0:
                additional.append((g, best_e, best_prob))
        
        return alignments + additional
    
    def _is_greek_function_word(self, word):
        """Check if a Greek word is likely a function word (article, particle)"""
        # Common Greek articles and particles
        function_words = {
            'ὁ', 'ἡ', 'τό', 'τοῦ', 'τῆς', 'τῷ', 'τήν', 'τόν', 'τῇ',
            'οἱ', 'αἱ', 'τά', 'τῶν', 'τοῖς', 'ταῖς', 'τούς', 'τάς',
            'καί', 'δέ', 'γάρ', 'οὖν', 'ἀλλά', 'τε', 'μέν',
            'ἐν', 'εἰς', 'ἐκ', 'ἀπό', 'πρός', 'διά', 'κατά', 'μετά',
            'ὅτι', 'ὡς', 'εἰ', 'ἵνα', 'ὅς', 'ἥ', 'ὅ',
        }
        # Normalize and check
        normalized = word.lower().strip()
        return normalized in function_words


def generate_interlinear(
    greek_text: str, 
    english_text: str, 
    alignments: List[Tuple[int, int, float]],
    show_confidence: bool = False,
    show_unaligned_english: bool = True,
):
    """Generate a beautiful interlinear display"""
    greek_words = greek_text.split()
    english_words = english_text.split()
    
    # Create alignment dict (Greek -> list of English words)
    greek_to_english = defaultdict(list)
    for g_idx, e_idx, conf in alignments:
        if 0 <= g_idx < len(greek_words) and 0 <= e_idx < len(english_words):
            greek_to_english[g_idx].append((english_words[e_idx], conf))
    
    # Find unaligned English words
    aligned_english = set(e_idx for _, e_idx, _ in alignments if 0 <= e_idx < len(english_words))
    unaligned_english = [
        (i, word) for i, word in enumerate(english_words) 
        if i not in aligned_english
    ]
    
    # Print header
    print("\n" + "═" * 80)
    print("📜 INTERLINEAR DISPLAY")
    print("═" * 80)
    
    # Print Greek text
    print(f"\n🇬🇷 Greek:   {greek_text}")
    print(f"🇬🇧 English: {english_text}")
    print()
    
    # Print word-by-word alignment
    print("─" * 80)
    print(f"{'#':<4} {'Greek Word':<20} {'→':<3} {'English Alignment':<40}")
    print("─" * 80)
    
    for i, greek_word in enumerate(greek_words):
        aligned_english_words = greek_to_english.get(i, [])
        
        if aligned_english_words:
            if show_confidence:
                english_str = ", ".join(
                    f"{word} ({conf:.2f})" for word, conf in aligned_english_words
                )
            else:
                english_str = ", ".join(word for word, _ in aligned_english_words)
        else:
            english_str = "⚠️ [no alignment]"
        
        print(f"{i+1:<4} {greek_word:<20} {'→':<3} {english_str:<40}")
    
    print("─" * 80)
    
    # Show unaligned English words
    if show_unaligned_english and unaligned_english:
        print(f"\n⚠️  Unaligned English words: ", end="")
        print(", ".join(f"'{word}' (pos {i})" for i, word in unaligned_english))
    
    print("═" * 80)


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
    test_file = Path('../data/test.json')
    if not test_file.exists():
        test_file = Path('data/test.json')
    if not test_file.exists():
        return None, None
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    for verse_data in data['data']:
        if (verse_data['book_name'] == book_name and
            verse_data['chapter'] == chapter and
            verse_data['verse'] == verse):
            
            greek_tokens = verse_data['greek_tokens']
            english_tokens = verse_data['english_tokens']
            
            greek_text = ' '.join(
                t['word'] if isinstance(t, dict) else t for t in greek_tokens
            )
            english_text = ' '.join(
                t['word'] if isinstance(t, dict) else t for t in english_tokens
            )
            
            return greek_text, english_text
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Test improved alignment model (v2)')
    parser.add_argument('--model-dir', default='./alignment_model_v2_output')
    parser.add_argument('--greek', help='Greek text to align')
    parser.add_argument('--english', help='English text to align')
    parser.add_argument('--verse', help='Verse reference (e.g., "John 1:1")')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Confidence threshold (uses optimal if not set)')
    parser.add_argument('--strategy', choices=['greedy', 'best_match', 'all'],
                       default='greedy', help='Alignment selection strategy')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Show confidence scores')
    parser.add_argument('--no-coverage', action='store_true',
                       help='Disable coverage post-processing')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    args = parser.parse_args()
    
    print("═" * 80)
    print("🔤 IMPROVED WORD ALIGNMENT INFERENCE (v2)")
    print("═" * 80)
    
    # Load model
    try:
        predictor = ImprovedAlignmentPredictor(args.model_dir)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nHave you trained the model yet?")
        print("Run: python train_alignment_v2.py")
        return
    
    # Interactive mode
    if args.interactive:
        print("\n📝 Interactive Mode - Enter Greek and English text to align")
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
                    alignments = predictor.align(
                        greek, english,
                        threshold=args.threshold,
                        strategy=args.strategy,
                        ensure_coverage=not args.no_coverage
                    )
                    generate_interlinear(
                        greek, english, alignments, args.show_confidence
                    )
                    print(f"\n✓ Found {len(alignments)} alignments\n")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
        return
    
    # Get text
    if args.verse:
        greek, english = load_verse_from_data(args.verse)
        if greek is None:
            print(f"\n❌ Verse not found: {args.verse}")
            print("\nTry one of these:")
            print("  - 'Jude 1'")
            print("  - '3 John 1'")
            print("  - 'Revelation 1:1'")
            return
        print(f"\n📖 Verse: {args.verse}")
    elif args.greek and args.english:
        greek = args.greek
        english = args.english
    else:
        # Default examples - John 1:1-4
        examples = [
            ("Ἐν ἀρχῇ ἦν ὁ λόγος", "In the beginning was the Word"),
            ("καὶ ὁ λόγος ἦν πρὸς τὸν θεόν", "and the Word was with God"),
            ("καὶ θεὸς ἦν ὁ λόγος", "and the Word was God"),
            ("πάντα δι' αὐτοῦ ἐγένετο", "All things were made through him"),
        ]
        
        print("\n🧪 Testing on example verses from John 1...")
        for greek, english in examples:
            print(f"\n📖 Greek:   {greek}")
            print(f"   English: {english}")
            alignments = predictor.align(
                greek, english,
                threshold=args.threshold,
                strategy=args.strategy,
                ensure_coverage=not args.no_coverage
            )
            generate_interlinear(
                greek, english, alignments, args.show_confidence
            )
        return
    
    # Predict alignments
    print(f"\n📖 Greek:   {greek}")
    print(f"   English: {english}")
    
    threshold_str = f"{args.threshold}" if args.threshold else f"{predictor.threshold} (optimal)"
    print(f"\n⚙️  Predicting alignments...")
    print(f"   Threshold: {threshold_str}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Coverage: {not args.no_coverage}")
    
    alignments = predictor.align(
        greek, english,
        threshold=args.threshold,
        strategy=args.strategy,
        ensure_coverage=not args.no_coverage
    )
    
    # Display
    generate_interlinear(greek, english, alignments, args.show_confidence)
    
    print(f"\n✓ Found {len(alignments)} alignments")
    
    # Show detailed alignment list
    if alignments:
        print("\n📋 Alignment pairs (sorted by confidence):")
        greek_words = greek.split()
        english_words = english.split()
        for g_idx, e_idx, conf in alignments[:15]:
            g_word = greek_words[g_idx] if g_idx < len(greek_words) else "?"
            e_word = english_words[e_idx] if e_idx < len(english_words) else "?"
            print(f"   {g_word:15s} → {e_word:15s} (conf: {conf:.3f})")
        if len(alignments) > 15:
            print(f"   ... and {len(alignments) - 15} more")


if __name__ == '__main__':
    main()

