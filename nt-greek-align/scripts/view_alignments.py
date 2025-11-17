#!/usr/bin/env python3
"""
View Greek-English Word Alignments

Display alignment pairs in a human-readable format.

Usage:
    python view_alignments.py [--file FILE] [--verse VERSE] [--limit N]

Examples:
    python view_alignments.py --verse "John 1:1"
    python view_alignments.py --verse "Romans 3:23" --file ../data/processed/alignments_nt_full.json
    python view_alignments.py --limit 5
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def format_alignment_table(verse_data: Dict) -> str:
    """Format a verse alignment as a readable table"""
    output = []
    
    # Header
    ref = f"{verse_data['book_name']} {verse_data['chapter']}:{verse_data['verse']}"
    output.append("=" * 80)
    output.append(f"  {ref}")
    output.append("=" * 80)
    
    # Greek text
    greek_words = [t['word'] for t in verse_data['greek_tokens']]
    output.append(f"\nGreek:   {' '.join(greek_words)}")
    
    # English text
    english_words = [t['word'] for t in verse_data['english_tokens']]
    output.append(f"English: {' '.join(english_words)}")
    
    # Alignment pairs
    if verse_data['alignments']:
        output.append(f"\nAlignments ({len(verse_data['alignments'])} pairs):")
        output.append("-" * 80)
        
        for i, align in enumerate(verse_data['alignments'][:20], 1):  # Limit to first 20
            greek_word = align['greek_word']
            greek_lemma = align['greek_lemma']
            english_word = align['english_word']
            strongs = align['strongs']
            
            output.append(f"  {i:2d}. {greek_word:15s} ({greek_lemma:12s}) → {english_word:15s} [{strongs}]")
        
        if len(verse_data['alignments']) > 20:
            output.append(f"  ... and {len(verse_data['alignments']) - 20} more alignments")
    else:
        output.append("\nNo alignments found for this verse.")
    
    output.append("")
    return "\n".join(output)


def format_alignment_visual(verse_data: Dict) -> str:
    """Format alignment as a visual diagram"""
    output = []
    
    # Header
    ref = f"{verse_data['book_name']} {verse_data['chapter']}:{verse_data['verse']}"
    output.append("=" * 80)
    output.append(f"  {ref} - Visual Alignment")
    output.append("=" * 80)
    
    greek_tokens = verse_data['greek_tokens']
    english_tokens = verse_data['english_tokens']
    alignments = verse_data['alignments']
    
    # Create alignment matrix
    output.append("\nGreek words (vertical) × English words (horizontal):")
    output.append("\n" + " " * 15 + "  ".join(f"{i:2d}" for i in range(min(10, len(english_tokens)))))
    
    for g_idx, g_token in enumerate(greek_tokens[:10]):  # Limit to first 10
        row = f"{g_idx:2d} {g_token['word'][:12]:12s} "
        for e_idx in range(min(10, len(english_tokens))):
            # Check if this pair is aligned
            is_aligned = any(
                a['greek_idx'] == g_idx and a['english_idx'] == e_idx 
                for a in alignments
            )
            row += " ● " if is_aligned else " · "
        output.append(row)
    
    output.append("\nEnglish words:")
    for e_idx, e_token in enumerate(english_tokens[:10]):
        output.append(f"  {e_idx:2d}. {e_token['word']}")
    
    if len(greek_tokens) > 10 or len(english_tokens) > 10:
        output.append("\n(Showing first 10 words only)")
    
    output.append("")
    return "\n".join(output)


def find_verse(data: Dict, verse_ref: str) -> Optional[Dict]:
    """Find a verse by reference (e.g., 'John 1:1' or 'Romans 3:23')"""
    # Parse reference
    parts = verse_ref.split()
    if len(parts) < 2:
        return None
    
    book_name = ' '.join(parts[:-1])
    chapter_verse = parts[-1]
    
    if ':' not in chapter_verse:
        return None
    
    try:
        chapter, verse = map(int, chapter_verse.split(':'))
    except ValueError:
        return None
    
    # Search for verse
    for verse_data in data['alignments']:
        if (verse_data['book_name'].lower() == book_name.lower() and
            verse_data['chapter'] == chapter and
            verse_data['verse'] == verse):
            return verse_data
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='View Greek-English word alignments'
    )
    parser.add_argument(
        '--file',
        default='../data/processed/alignments_nt_full.json',
        help='Alignment JSON file to read'
    )
    parser.add_argument(
        '--verse',
        help='Specific verse to display (e.g., "John 1:1")'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1,
        help='Number of verses to display (if --verse not specified)'
    )
    parser.add_argument(
        '--visual',
        action='store_true',
        help='Show visual alignment matrix'
    )
    
    args = parser.parse_args()
    
    # Load alignment data
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Loading alignments from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Display metadata
    print("\n" + "=" * 80)
    print("ALIGNMENT DATASET SUMMARY")
    print("=" * 80)
    for key, value in data['metadata'].items():
        print(f"  {key:25s}: {value}")
    print("=" * 80)
    
    # Display verses
    if args.verse:
        verse_data = find_verse(data, args.verse)
        if verse_data:
            if args.visual:
                print(format_alignment_visual(verse_data))
            print(format_alignment_table(verse_data))
        else:
            print(f"\nVerse not found: {args.verse}")
            print("\nTry one of these:")
            print("  - 'John 1:1'")
            print("  - 'Romans 3:23'")
            print("  - 'Matthew 5:3'")
    else:
        # Display first N verses
        for i, verse_data in enumerate(data['alignments'][:args.limit]):
            if args.visual:
                print(format_alignment_visual(verse_data))
            print(format_alignment_table(verse_data))


if __name__ == '__main__':
    main()

