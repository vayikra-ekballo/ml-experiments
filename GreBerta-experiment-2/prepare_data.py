#!/usr/bin/env python3
"""
Prepare Train/Dev/Test Splits for Fine-Tuning

Creates stratified splits of the NT alignment dataset by books.

Usage:
    python prepare_data.py
"""

import json
from pathlib import Path
from collections import Counter


def main():
    print("=" * 80)
    print("Preparing Train/Dev/Test Splits for GreBerta Fine-Tuning")
    print("=" * 80)
    
    # Load full NT alignment dataset
    data_path = Path('../nt-greek-align/data/processed/alignments_nt_full.json')
    
    if not data_path.exists():
        print(f"\nError: Alignment file not found at {data_path}")
        print("Please run extract_alignments.py first.")
        return
    
    print(f"\nLoading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data['alignments'])} verses")
    print(f"  Metadata: {data['metadata']}")
    
    # Define splits by books
    # Train: ~80% (20 books)
    # Dev: ~15% (4 books) 
    # Test: ~5% (3 books)
    
    train_books = [
        'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
        '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
        'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
        '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James'
    ]
    
    dev_books = [
        '1 Peter', '2 Peter', '1 John', '2 John'
    ]
    
    test_books = [
        '3 John', 'Jude', 'Revelation'
    ]
    
    print("\nSplit configuration:")
    print(f"  Train: {len(train_books)} books - {train_books[:3]}... ")
    print(f"  Dev:   {len(dev_books)} books - {dev_books}")
    print(f"  Test:  {len(test_books)} books - {test_books}")
    
    # Split data by books
    train_data = [v for v in data['alignments'] if v['book_name'] in train_books]
    dev_data = [v for v in data['alignments'] if v['book_name'] in dev_books]
    test_data = [v for v in data['alignments'] if v['book_name'] in test_books]
    
    # Statistics
    def get_stats(verses, name):
        total_verses = len(verses)
        total_greek_tokens = sum(len(v['greek_tokens']) for v in verses)
        total_english_tokens = sum(len(v['english_tokens']) for v in verses)
        total_alignments = sum(len(v['alignments']) for v in verses)
        verses_with_alignments = sum(1 for v in verses if v['alignments'])
        
        print(f"\n{name} Statistics:")
        print(f"  Verses: {total_verses:,}")
        print(f"  Greek tokens: {total_greek_tokens:,}")
        print(f"  English tokens: {total_english_tokens:,}")
        print(f"  Alignment pairs: {total_alignments:,}")
        print(f"  Coverage: {verses_with_alignments}/{total_verses} ({verses_with_alignments/total_verses*100:.1f}%)")
        
        return {
            'verses': total_verses,
            'greek_tokens': total_greek_tokens,
            'english_tokens': total_english_tokens,
            'alignments': total_alignments,
            'coverage': f"{verses_with_alignments/total_verses*100:.1f}%"
        }
    
    train_stats = get_stats(train_data, "Train")
    dev_stats = get_stats(dev_data, "Dev")
    test_stats = get_stats(test_data, "Test")
    
    # Show book distribution
    print("\nBook distribution:")
    for split_name, split_data in [('Train', train_data), ('Dev', dev_data), ('Test', test_data)]:
        book_counts = Counter(v['book_name'] for v in split_data)
        print(f"\n  {split_name}:")
        for book, count in sorted(book_counts.items()):
            print(f"    {book:20s}: {count:4d} verses")
    
    # Save splits
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}/...")
    
    # Save train
    train_output = {
        'metadata': {
            **data['metadata'],
            'split': 'train',
            'books': train_books,
            **train_stats
        },
        'data': train_data
    }
    with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {output_dir / 'train.json'} ({len(train_data)} verses)")
    
    # Save dev
    dev_output = {
        'metadata': {
            **data['metadata'],
            'split': 'dev',
            'books': dev_books,
            **dev_stats
        },
        'data': dev_data
    }
    with open(output_dir / 'dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev_output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {output_dir / 'dev.json'} ({len(dev_data)} verses)")
    
    # Save test
    test_output = {
        'metadata': {
            **data['metadata'],
            'split': 'test',
            'books': test_books,
            **test_stats
        },
        'data': test_data
    }
    with open(output_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {output_dir / 'test.json'} ({len(test_data)} verses)")
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nTrain/Dev/Test splits saved to {output_dir}/")
    print("\nNext steps:")
    print("  1. Examine splits: python -m json.tool data/train.json | head -100")
    print("  2. Choose task: POS tagging or word alignment")
    print("  3. Implement training script")
    print("=" * 80)


if __name__ == '__main__':
    main()

