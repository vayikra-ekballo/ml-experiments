#!/usr/bin/env python3
"""
Compare v1 and v2 alignment models on example verses

This script helps visualize the improvement from v1 to v2 models
by showing side-by-side alignment results.

Usage:
    python compare_models.py
    python compare_models.py --v1-dir ./alignment_model_output --v2-dir ./alignment_model_v2_output
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

# Try to import both models
try:
    from train_alignment import AlignmentModel as AlignmentModelV1
    from test_alignment import AlignmentPredictor as PredictorV1
    V1_AVAILABLE = True
except ImportError:
    V1_AVAILABLE = False
    print("⚠️  V1 model not available")

try:
    from train_alignment_v2 import ImprovedAlignmentModel as AlignmentModelV2
    from test_alignment_v2 import ImprovedAlignmentPredictor as PredictorV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    print("⚠️  V2 model not available")


# Test examples
EXAMPLES = [
    {
        "name": "John 1:1a",
        "greek": "Ἐν ἀρχῇ ἦν ὁ λόγος",
        "english": "In the beginning was the Word",
        "expected": {
            "Ἐν": "In",
            "ἀρχῇ": "beginning",
            "ἦν": "was",
            "ὁ": "the",
            "λόγος": "Word",
        }
    },
    {
        "name": "John 1:1b",
        "greek": "καὶ ὁ λόγος ἦν πρὸς τὸν θεόν",
        "english": "and the Word was with God",
        "expected": {
            "καὶ": "and",
            "ὁ": "the",
            "λόγος": "Word",
            "ἦν": "was",
            "πρὸς": "with",
            "τὸν": None,  # Article
            "θεόν": "God",
        }
    },
    {
        "name": "John 1:1c",
        "greek": "καὶ θεὸς ἦν ὁ λόγος",
        "english": "and the Word was God",
        "expected": {
            "καὶ": "and",
            "θεὸς": "God",
            "ἦν": "was",
            "ὁ": "the",
            "λόγος": "Word",
        }
    },
    {
        "name": "John 1:3a",
        "greek": "πάντα δι' αὐτοῦ ἐγένετο",
        "english": "All things were made through him",
        "expected": {
            "πάντα": "All things",
            "δι'": "through",
            "αὐτοῦ": "him",
            "ἐγένετο": "were made",
        }
    },
]


def format_alignments(greek_text: str, alignments: List[Tuple[int, int, float]]) -> dict:
    """Convert alignments to Greek word -> English words mapping"""
    greek_words = greek_text.split()
    result = {}
    
    for g_idx, e_idx, conf in alignments:
        if 0 <= g_idx < len(greek_words):
            g_word = greek_words[g_idx]
            if g_word not in result:
                result[g_word] = []
            result[g_word].append((e_idx, conf))
    
    return result


def evaluate_alignments(
    greek_text: str, 
    english_text: str, 
    alignments: List[Tuple[int, int, float]], 
    expected: dict
) -> dict:
    """Evaluate alignment quality against expected alignments"""
    greek_words = greek_text.split()
    english_words = english_text.split()
    
    correct = 0
    total_expected = 0
    false_positives = 0
    
    aligned_map = {}
    for g_idx, e_idx, conf in alignments:
        if 0 <= g_idx < len(greek_words) and 0 <= e_idx < len(english_words):
            g_word = greek_words[g_idx]
            e_word = english_words[e_idx]
            if g_word not in aligned_map:
                aligned_map[g_word] = set()
            aligned_map[g_word].add(e_word)
    
    for g_word, expected_english in expected.items():
        if expected_english is None:
            continue  # Skip function words
        
        total_expected += 1
        predicted = aligned_map.get(g_word, set())
        
        if isinstance(expected_english, str):
            expected_english = {expected_english}
        else:
            expected_english = set(expected_english.split())
        
        if predicted & expected_english:
            correct += 1
    
    accuracy = correct / total_expected if total_expected > 0 else 0
    
    return {
        "correct": correct,
        "total": total_expected,
        "accuracy": accuracy,
    }


def print_comparison(example: dict, v1_alignments: list, v2_alignments: list, english_text: str):
    """Print side-by-side comparison"""
    greek_words = example["greek"].split()
    english_words = english_text.split()
    
    # Build alignment maps
    v1_map = {}
    for g_idx, e_idx, conf in v1_alignments:
        if 0 <= g_idx < len(greek_words):
            g_word = greek_words[g_idx]
            if g_word not in v1_map:
                v1_map[g_word] = []
            if 0 <= e_idx < len(english_words):
                v1_map[g_word].append(english_words[e_idx])
    
    v2_map = {}
    for g_idx, e_idx, conf in v2_alignments:
        if 0 <= g_idx < len(greek_words):
            g_word = greek_words[g_idx]
            if g_word not in v2_map:
                v2_map[g_word] = []
            if 0 <= e_idx < len(english_words):
                v2_map[g_word].append(english_words[e_idx])
    
    print(f"\n{'Greek':<15} {'Expected':<20} {'V1 Model':<20} {'V2 Model':<20}")
    print("-" * 75)
    
    for g_word in greek_words:
        expected = example["expected"].get(g_word, "?")
        if expected is None:
            expected = "(func)"
        
        v1_result = ", ".join(v1_map.get(g_word, ["—"]))
        v2_result = ", ".join(v2_map.get(g_word, ["—"]))
        
        # Mark correct/incorrect
        v1_mark = "✓" if expected != "?" and expected != "(func)" and expected in v1_result else ""
        v2_mark = "✓" if expected != "?" and expected != "(func)" and expected in v2_result else ""
        
        print(f"{g_word:<15} {expected:<20} {v1_result:<18}{v1_mark} {v2_result:<18}{v2_mark}")


def main():
    parser = argparse.ArgumentParser(description="Compare v1 and v2 alignment models")
    parser.add_argument("--v1-dir", default="./alignment_model_output")
    parser.add_argument("--v2-dir", default="./alignment_model_v2_output")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    
    print("=" * 75)
    print("🔬 ALIGNMENT MODEL COMPARISON: V1 vs V2")
    print("=" * 75)
    
    # Load models
    v1_predictor = None
    v2_predictor = None
    
    if V1_AVAILABLE:
        try:
            v1_predictor = PredictorV1(args.v1_dir)
        except Exception as e:
            print(f"\n⚠️  Could not load V1 model: {e}")
    
    if V2_AVAILABLE:
        try:
            v2_predictor = PredictorV2(args.v2_dir)
        except Exception as e:
            print(f"\n⚠️  Could not load V2 model: {e}")
    
    if v1_predictor is None and v2_predictor is None:
        print("\n❌ No models available for comparison!")
        print("\nPlease train at least one model:")
        print("  python train_alignment.py     # V1")
        print("  python train_alignment_v2.py  # V2")
        return
    
    # Compare on examples
    v1_total_score = 0
    v2_total_score = 0
    
    for example in EXAMPLES:
        print(f"\n{'=' * 75}")
        print(f"📖 {example['name']}")
        print(f"   Greek:   {example['greek']}")
        print(f"   English: {example['english']}")
        
        # Get alignments from both models
        v1_alignments = []
        v2_alignments = []
        
        if v1_predictor:
            threshold = args.threshold if args.threshold else 0.5
            v1_alignments = v1_predictor.align(
                example["greek"], example["english"], threshold
            )
        
        if v2_predictor:
            v2_alignments = v2_predictor.align(
                example["greek"], example["english"],
                threshold=args.threshold,
                strategy='greedy',
                ensure_coverage=True
            )
        
        # Print comparison
        print_comparison(example, v1_alignments, v2_alignments, example["english"])
        
        # Evaluate
        if v1_predictor:
            v1_eval = evaluate_alignments(
                example["greek"], example["english"], v1_alignments, example["expected"]
            )
            v1_total_score += v1_eval["accuracy"]
            print(f"\n   V1 Score: {v1_eval['correct']}/{v1_eval['total']} ({v1_eval['accuracy']:.1%})")
        
        if v2_predictor:
            v2_eval = evaluate_alignments(
                example["greek"], example["english"], v2_alignments, example["expected"]
            )
            v2_total_score += v2_eval["accuracy"]
            print(f"   V2 Score: {v2_eval['correct']}/{v2_eval['total']} ({v2_eval['accuracy']:.1%})")
    
    # Summary
    num_examples = len(EXAMPLES)
    print(f"\n{'=' * 75}")
    print("📊 SUMMARY")
    print("=" * 75)
    
    if v1_predictor:
        print(f"\nV1 Model Average: {v1_total_score/num_examples:.1%}")
    if v2_predictor:
        print(f"V2 Model Average: {v2_total_score/num_examples:.1%}")
    
    if v1_predictor and v2_predictor:
        improvement = (v2_total_score - v1_total_score) / num_examples
        print(f"\nImprovement: {improvement:+.1%}")
        
        if improvement > 0:
            print("✅ V2 model shows improvement!")
        elif improvement < 0:
            print("⚠️  V2 model appears worse - may need more training")
        else:
            print("➡️  Models perform similarly")
    
    print("=" * 75)


if __name__ == "__main__":
    main()

