#!/usr/bin/env python3
"""
Extract Greek-English Word Alignments from SBLGNT and WEB

This script parses the SBLGNT Greek New Testament (with morphological annotations)
and the World English Bible (with Strong's numbers) to create word-level alignment
pairs for training alignment models.

Usage:
    python extract_alignments.py [--book BOOK] [--output OUTPUT]

Example:
    python extract_alignments.py --book 64-Jn-morphgnt.txt --output alignments_john.json
    python extract_alignments.py --output alignments_full_nt.json
"""

import re
import json
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# Book mapping: SBLGNT filename code -> WEB filename code
# SBLGNT files: 61-Mt-morphgnt.txt, 64-Jn-morphgnt.txt, etc.
# WEB files: 70-MATeng-web.usfm, 73-JHNeng-web.usfm, etc.
FILENAME_TO_WEB = {
    '61': '70',  # Matthew
    '62': '71',  # Mark
    '63': '72',  # Luke
    '64': '73',  # John
    '65': '74',  # Acts
    '66': '75',  # Romans
    '67': '76',  # 1 Corinthians
    '68': '77',  # 2 Corinthians
    '69': '78',  # Galatians
    '70': '79',  # Ephesians
    '71': '80',  # Philippians
    '72': '81',  # Colossians
    '73': '82',  # 1 Thessalonians
    '74': '83',  # 2 Thessalonians
    '75': '84',  # 1 Timothy
    '76': '85',  # 2 Timothy
    '77': '86',  # Titus
    '78': '87',  # Philemon
    '79': '88',  # Hebrews
    '80': '89',  # James
    '81': '90',  # 1 Peter
    '82': '91',  # 2 Peter
    '83': '92',  # 1 John
    '84': '93',  # 2 John
    '85': '94',  # 3 John
    '86': '95',  # Jude
    '87': '96',  # Revelation
}

# Book names by filename prefix
# SBLGNT filenames: 61-Mt-morphgnt.txt, 64-Jn-morphgnt.txt, etc.
FILENAME_TO_BOOKNAME = {
    '61': 'Matthew', '62': 'Mark', '63': 'Luke', '64': 'John',
    '65': 'Acts', '66': 'Romans', '67': '1 Corinthians', '68': '2 Corinthians',
    '69': 'Galatians', '70': 'Ephesians', '71': 'Philippians', '72': 'Colossians',
    '73': '1 Thessalonians', '74': '2 Thessalonians', '75': '1 Timothy', '76': '2 Timothy',
    '77': 'Titus', '78': 'Philemon', '79': 'Hebrews', '80': 'James',
    '81': '1 Peter', '82': '2 Peter', '83': '1 John', '84': '2 John',
    '85': '3 John', '86': 'Jude', '87': 'Revelation'
}

# Book names by verse_id prefix (used inside SBLGNT files)
# Verse IDs in files: 040101 = book 04 (John), chapter 01, verse 01
VERSEID_TO_BOOKNAME = {
    '01': 'Matthew', '02': 'Mark', '03': 'Luke', '04': 'John',
    '05': 'Acts', '06': 'Romans', '07': '1 Corinthians', '08': '2 Corinthians',
    '09': 'Galatians', '10': 'Ephesians', '11': 'Philippians', '12': 'Colossians',
    '13': '1 Thessalonians', '14': '2 Thessalonians', '15': '1 Timothy', '16': '2 Timothy',
    '17': 'Titus', '18': 'Philemon', '19': 'Hebrews', '20': 'James',
    '21': '1 Peter', '22': '2 Peter', '23': '1 John', '24': '2 John',
    '25': '3 John', '26': 'Jude', '27': 'Revelation'
}


class GreekToken:
    """Represents a Greek word token from SBLGNT"""
    def __init__(self, verse_id: str, pos: str, parsing: str, 
                 surface: str, word: str, normalized: str, lemma: str):
        self.verse_id = verse_id
        self.pos = pos
        self.parsing = parsing
        self.surface = surface
        self.word = word
        self.normalized = normalized
        self.lemma = unicodedata.normalize('NFC', lemma)  # Normalize for lookups
        
    def to_dict(self):
        return {
            'surface': self.surface,
            'word': self.word,
            'normalized': self.normalized,
            'lemma': self.lemma,
            'pos': self.pos,
            'parsing': self.parsing
        }
    
    def __repr__(self):
        return f"GreekToken({self.word}/{self.lemma})"


class EnglishToken:
    """Represents an English word token from WEB"""
    def __init__(self, word: str, strongs: Optional[str] = None):
        self.word = word
        self.strongs = strongs
        
    def to_dict(self):
        return {
            'word': self.word,
            'strongs': self.strongs
        }
    
    def __repr__(self):
        return f"EnglishToken({self.word}/{self.strongs})"


def parse_sblgnt_file(filepath: Path) -> Dict[str, List[GreekToken]]:
    """
    Parse an SBLGNT morphological Greek NT file
    
    Returns:
        Dict mapping verse_id -> list of GreekToken objects
    """
    verses = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split by whitespace (the file uses spaces, not tabs)
            parts = line.split()
            if len(parts) != 7:
                continue
                
            verse_id, pos, parsing, surface, word, normalized, lemma = parts
            
            token = GreekToken(verse_id, pos, parsing, surface, word, normalized, lemma)
            verses[verse_id].append(token)
    
    return verses


def parse_usfm_verse(verse_text: str) -> List[EnglishToken]:
    """
    Parse a USFM verse and extract English words with Strong's numbers
    
    Example input:
        \\w In|strong="G1722"\\w* \\w the|strong="G1722"\\w* beginning
    
    Returns:
        List of EnglishToken objects
    """
    tokens = []
    
    # Pattern to match: \w word|strong="G####"\w*
    pattern = r'\\w\s+([^|\\]+?)(?:\|strong="(G\d+)")?\s*\\w\*'
    
    for match in re.finditer(pattern, verse_text):
        word = match.group(1).strip()
        strongs = match.group(2)  # May be None
        
        # Clean up word (remove possessives, etc.)
        word = word.replace("'s", '').strip()
        
        if word:  # Only add non-empty words
            tokens.append(EnglishToken(word, strongs))
    
    # Also capture words without Strong's numbers (plain text)
    # Remove all \w ... \w* patterns first
    plain_text = re.sub(r'\\w\s+[^\\]+?\\w\*', '', verse_text)
    # Remove other USFM markers
    plain_text = re.sub(r'\\[a-z]+\d*\s*', '', plain_text)
    # Remove footnotes and cross-references
    plain_text = re.sub(r'\\[fx]\s+.*?\\[fx]\*', '', plain_text)
    
    # Extract remaining words
    for word in plain_text.split():
        word = word.strip('.,;:!?"\'()[]')
        if word and len(word) > 0:
            tokens.append(EnglishToken(word, None))
    
    return tokens


def parse_usfm_file(filepath: Path) -> Dict[str, List[EnglishToken]]:
    """
    Parse a USFM English Bible file
    
    Returns:
        Dict mapping verse_id -> list of EnglishToken objects
    """
    verses = {}
    current_chapter = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Match chapter marker: \c 1
            chapter_match = re.match(r'\\c\s+(\d+)', line)
            if chapter_match:
                current_chapter = int(chapter_match.group(1))
                continue
            
            # Match verse marker: \v 1 ...verse text...
            verse_match = re.match(r'\\v\s+(\d+)\s+(.+)', line)
            if verse_match and current_chapter is not None:
                verse_num = int(verse_match.group(1))
                verse_text = verse_match.group(2)
                
                # Create verse_id in SBLGNT format (BBCCVV)
                # Need to extract book code from filename
                tokens = parse_usfm_verse(verse_text)
                verses[(current_chapter, verse_num)] = tokens
    
    return verses


def load_lemma_to_strongs(filepath: Path) -> Dict[str, List[str]]:
    """Load the lemma -> Strong's numbers mapping"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize all Greek lemmas to NFC
    normalized_data = {}
    for lemma, strongs_list in data.items():
        normalized_lemma = unicodedata.normalize('NFC', lemma)
        normalized_data[normalized_lemma] = strongs_list
    
    return normalized_data


def create_alignments(greek_verses: Dict[str, List[GreekToken]],
                     english_verses: Dict[Tuple[int, int], List[EnglishToken]],
                     lemma_to_strongs: Dict[str, List[str]],
                     book_code: str) -> List[Dict]:
    """
    Create alignment pairs between Greek and English tokens
    
    Returns:
        List of alignment dictionaries
    """
    alignments = []
    
    for verse_id, greek_tokens in greek_verses.items():
        # Parse verse_id: BBCCVV format
        if len(verse_id) != 6:
            continue
            
        book = verse_id[:2]
        chapter = int(verse_id[2:4])
        verse = int(verse_id[4:6])
        
        # Get corresponding English verse
        english_tokens = english_verses.get((chapter, verse))
        if english_tokens is None:
            continue
        
        # Create verse-level alignment entry
        verse_alignment = {
            'book_code': book_code,
            'book_name': VERSEID_TO_BOOKNAME.get(book, 'Unknown'),
            'chapter': chapter,
            'verse': verse,
            'verse_id': verse_id,
            'greek_tokens': [t.to_dict() for t in greek_tokens],
            'english_tokens': [t.to_dict() for t in english_tokens],
            'alignments': []
        }
        
        # Create word-level alignments using Strong's numbers
        for greek_idx, greek_token in enumerate(greek_tokens):
            # Get Strong's number(s) for this Greek lemma
            strongs_numbers = lemma_to_strongs.get(greek_token.lemma, [])
            
            if not strongs_numbers:
                # Try without final sigma
                alt_lemma = greek_token.lemma.replace('ς', 'σ')
                strongs_numbers = lemma_to_strongs.get(alt_lemma, [])
            
            # Find English tokens with matching Strong's numbers
            for english_idx, english_token in enumerate(english_tokens):
                if english_token.strongs in strongs_numbers:
                    verse_alignment['alignments'].append({
                        'greek_idx': greek_idx,
                        'english_idx': english_idx,
                        'greek_word': greek_token.word,
                        'greek_lemma': greek_token.lemma,
                        'english_word': english_token.word,
                        'strongs': english_token.strongs,
                        'confidence': 'strongs_match'  # Indicates this is from Strong's
                    })
        
        alignments.append(verse_alignment)
    
    return alignments


def format_verse_id(chapter: int, verse: int, book_code: str) -> str:
    """Format chapter/verse into SBLGNT verse_id format (BBCCVV)"""
    return f"{book_code}{chapter:02d}{verse:02d}"


def main():
    parser = argparse.ArgumentParser(
        description='Extract Greek-English word alignments from SBLGNT and WEB'
    )
    parser.add_argument(
        '--book',
        help='Specific SBLGNT book file to process (e.g., 64-Jn-morphgnt.txt)',
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output JSON file path',
        default='alignments_nt.json'
    )
    parser.add_argument(
        '--data-dir',
        help='Base data directory',
        default='../data'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path(__file__).parent.parent / 'data'
    if args.data_dir:
        base_dir = Path(args.data_dir)
    
    greek_dir = base_dir / 'raw' / 'greek' / 'sblgnt'
    english_dir = base_dir / 'raw' / 'english'
    lexicon_file = base_dir / 'processed' / 'lemma_to_strongs.json'
    
    # Load lexicon
    print("Loading lemma -> Strong's mapping...")
    lemma_to_strongs = load_lemma_to_strongs(lexicon_file)
    print(f"  Loaded {len(lemma_to_strongs)} lemmas")
    
    # Determine which books to process
    if args.book:
        greek_files = [greek_dir / args.book]
    else:
        greek_files = sorted(greek_dir.glob('*-morphgnt.txt'))
    
    all_alignments = []
    stats = {
        'total_verses': 0,
        'total_greek_tokens': 0,
        'total_english_tokens': 0,
        'total_alignments': 0,
        'verses_with_alignments': 0
    }
    
    # Process each book
    for greek_file in greek_files:
        # Extract book code from filename (e.g., "64" from "64-Jn-morphgnt.txt")
        book_code = greek_file.name.split('-')[0]
        book_name = FILENAME_TO_BOOKNAME.get(book_code, 'Unknown')
        
        print(f"\nProcessing {book_name} (filename code: {book_code})...")
        
        # Find corresponding English file
        web_code = FILENAME_TO_WEB.get(book_code)
        if not web_code:
            print(f"  Warning: No WEB mapping found for book code {book_code}")
            continue
        
        # English files use pattern like "73-JHNeng-web.usfm"
        english_file = None
        for ef in english_dir.glob(f'{web_code}-*eng-web.usfm'):
            english_file = ef
            break
        
        if not english_file or not english_file.exists():
            print(f"  Warning: English file not found for book code {web_code}")
            continue
        
        print(f"  Greek: {greek_file.name}")
        print(f"  English: {english_file.name}")
        
        # Parse files
        print("  Parsing Greek...")
        greek_verses = parse_sblgnt_file(greek_file)
        print(f"    Found {len(greek_verses)} verses")
        
        print("  Parsing English...")
        english_verses = parse_usfm_file(english_file)
        print(f"    Found {len(english_verses)} verses")
        
        # Create alignments
        print("  Creating alignments...")
        book_alignments = create_alignments(
            greek_verses, english_verses, lemma_to_strongs, book_code
        )
        
        # Update statistics
        for verse_align in book_alignments:
            stats['total_verses'] += 1
            stats['total_greek_tokens'] += len(verse_align['greek_tokens'])
            stats['total_english_tokens'] += len(verse_align['english_tokens'])
            stats['total_alignments'] += len(verse_align['alignments'])
            if verse_align['alignments']:
                stats['verses_with_alignments'] += 1
        
        print(f"    Created {len(book_alignments)} verse alignments")
        print(f"    Total alignment pairs: {sum(len(v['alignments']) for v in book_alignments)}")
        
        all_alignments.extend(book_alignments)
    
    # Save output
    output_path = Path(args.output)
    print(f"\nSaving alignments to {output_path}...")
    
    output_data = {
        'metadata': {
            'source_greek': 'SBLGNT with MorphGNT annotations',
            'source_english': 'World English Bible (WEB)',
            'alignment_method': 'Strong\'s concordance numbers',
            'total_books': len(set(a['book_code'] for a in all_alignments)),
            'total_verses': stats['total_verses'],
            'total_greek_tokens': stats['total_greek_tokens'],
            'total_english_tokens': stats['total_english_tokens'],
            'total_alignments': stats['total_alignments'],
            'verses_with_alignments': stats['verses_with_alignments'],
            'alignment_coverage': f"{stats['verses_with_alignments'] / stats['total_verses'] * 100:.1f}%" if stats['total_verses'] > 0 else "0%"
        },
        'alignments': all_alignments
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("ALIGNMENT EXTRACTION COMPLETE")
    print("="*70)
    print(f"Books processed:    {output_data['metadata']['total_books']}")
    print(f"Total verses:       {stats['total_verses']}")
    print(f"Greek tokens:       {stats['total_greek_tokens']}")
    print(f"English tokens:     {stats['total_english_tokens']}")
    print(f"Alignment pairs:    {stats['total_alignments']}")
    print(f"Verses w/ aligns:   {stats['verses_with_alignments']} ({output_data['metadata']['alignment_coverage']})")
    avg_per_verse = stats['total_alignments'] / stats['total_verses'] if stats['total_verses'] > 0 else 0
    print(f"Avg aligns/verse:   {avg_per_verse:.1f}")
    print(f"\nOutput saved to: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()

