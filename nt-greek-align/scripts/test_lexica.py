#!/usr/bin/env python3
"""
Test script to verify lexicon data is loaded correctly.
"""

import json
import unicodedata
from pathlib import Path

def load_lexica():
    """Load all lexicon files."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'data' / 'processed'
    
    with open(processed_dir / 'lemma_to_glosses.json', 'r', encoding='utf-8') as f:
        lemma_glosses_raw = json.load(f)
    
    with open(processed_dir / 'lemma_to_strongs.json', 'r', encoding='utf-8') as f:
        lemma_strongs_raw = json.load(f)
    
    with open(processed_dir / 'strongs_dictionary.json', 'r', encoding='utf-8') as f:
        strongs = json.load(f)
    
    # Normalize to NFC for consistent lookups
    lemma_glosses = {unicodedata.normalize('NFC', k): v for k, v in lemma_glosses_raw.items()}
    lemma_strongs = {unicodedata.normalize('NFC', k): v for k, v in lemma_strongs_raw.items()}
    
    return lemma_glosses, lemma_strongs, strongs

def lookup_lemma(lemma, lemma_glosses, lemma_strongs, strongs):
    """Look up a Greek lemma and display all available information."""
    lemma_nfc = unicodedata.normalize('NFC', lemma)
    
    if lemma_nfc not in lemma_glosses:
        return f"Lemma '{lemma}' not found in lexicon"
    
    glosses = lemma_glosses[lemma_nfc]
    strongs_nums = lemma_strongs.get(lemma_nfc, [])
    
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"Greek: {lemma}")
    
    if strongs_nums:
        strongs_num = strongs_nums[0]
        if strongs_num in strongs:
            data = strongs[strongs_num]
            output.append(f"Strong's: {strongs_num}")
            output.append(f"Translit: {data['translit']}")
            if data['strongs_def']:
                output.append(f"Definition: {data['strongs_def'][:100]}...")
    
    output.append(f"English Glosses ({len(glosses)}):")
    output.append(f"  {', '.join(glosses[:15])}")
    if len(glosses) > 15:
        output.append(f"  ... and {len(glosses) - 15} more")
    
    return '\n'.join(output)

def main():
    print("Loading lexica...")
    lemma_glosses, lemma_strongs, strongs = load_lexica()
    
    print(f"✓ Loaded {len(lemma_glosses)} Greek lemmas")
    print(f"✓ Loaded {len(strongs)} Strong's entries")
    
    # Test common NT words
    test_words = [
        ('λόγος', 'word, speech'),
        ('θεός', 'God'),
        ('ἀγάπη', 'love'),
        ('πιστεύω', 'believe'),
        ('ἄνθρωπος', 'man, human'),
        ('εἰμί', 'to be'),
        ('Ἰησοῦς', 'Jesus'),
        ('Χριστός', 'Christ'),
        ('πνεῦμα', 'spirit'),
        ('καρδία', 'heart'),
    ]
    
    print("\n" + "="*70)
    print("COMMON NEW TESTAMENT GREEK WORDS")
    print("="*70)
    
    for lemma, english_hint in test_words:
        result = lookup_lemma(lemma, lemma_glosses, lemma_strongs, strongs)
        print(result)
    
    print("\n" + "="*70)
    print("✓ Lexicon test complete!")
    print("="*70)

if __name__ == '__main__':
    main()

