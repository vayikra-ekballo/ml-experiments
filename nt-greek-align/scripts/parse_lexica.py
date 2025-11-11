#!/usr/bin/env python3
"""
Parse Greek lexica to create useful mappings for alignment.

Creates:
1. Strong's number -> lemma, gloss, definition
2. Lemma -> Strong's numbers, glosses
3. Lemma -> English gloss list (for dictionary priors)
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def parse_strongs_xml(xml_path: Path) -> Dict[str, Dict]:
    """
    Parse Strong's Greek dictionary XML.
    
    Returns dict: { 'G1234': { 'lemma': '...', 'translit': '...', 
                                'kjv_def': '...', 'strongs_def': '...' } }
    """
    print(f"Parsing Strong's dictionary from {xml_path}...")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    strongs_dict = {}
    
    for entry in root.findall('.//entry'):
        strongs_num = entry.get('strongs')
        if not strongs_num:
            continue
            
        # Get Strong's number in G#### format
        strongs_id = f"G{int(strongs_num)}"
        
        # Extract data
        lemma = None
        translit = None
        kjv_def = None
        strongs_def = None
        
        # Get Greek lemma
        greek_elem = entry.find('.//greek')
        if greek_elem is not None:
            lemma = greek_elem.get('unicode', '')
            translit = greek_elem.get('translit', '')
        
        # Get definitions
        kjv_elem = entry.find('.//kjv_def')
        if kjv_elem is not None and kjv_elem.text:
            kjv_def = kjv_elem.text.strip()
        
        strongs_elem = entry.find('.//strongs_def')
        if strongs_elem is not None and strongs_elem.text:
            strongs_def = strongs_elem.text.strip()
        
        if lemma:
            strongs_dict[strongs_id] = {
                'lemma': lemma,
                'translit': translit,
                'kjv_def': kjv_def,
                'strongs_def': strongs_def
            }
    
    print(f"  Parsed {len(strongs_dict)} Strong's entries")
    return strongs_dict


def extract_glosses_from_kjv_def(kjv_def: str) -> List[str]:
    """
    Extract individual English glosses from KJV definition string.
    
    Example: ":--account, cause, communication, X concerning, doctrine"
    Returns: ["account", "cause", "communication", "doctrine"]
    """
    if not kjv_def:
        return []
    
    # Remove :-- or -- prefix if present
    kjv_def = re.sub(r'^:?--', '', kjv_def).strip()
    
    # Remove newlines
    kjv_def = kjv_def.replace('\n', ' ')
    
    # Split by comma or semicolon
    parts = re.split(r'[,;]', kjv_def)
    
    glosses = []
    for part in parts:
        part = part.strip()
        
        # Remove "X " markers (indicate less common usage)
        part = re.sub(r'\bX\s+', '', part)
        
        # Remove parenthetical notes
        part = re.sub(r'\([^)]*\)', '', part)
        
        # Remove + markers
        part = re.sub(r'\+\s*', '', part)
        
        # Skip empty or very short
        part = part.strip()
        if len(part) < 2:
            continue
        
        # Skip if it's a period or just punctuation
        if re.match(r'^[^a-zA-Z]+$', part):
            continue
        
        # Skip if contains special chars indicating it's not a simple gloss
        if any(x in part for x in ['...', 'i.e.', 'see', 'from']):
            continue
        
        # Remove trailing period
        part = part.rstrip('.')
        
        # Clean up whitespace
        part = ' '.join(part.split())
        
        if not part:
            continue
        
        glosses.append(part.lower())
    
    # Clean up glosses further - split compound forms
    cleaned = []
    for gloss in glosses:
        # Handle hyphenated alternatives like "god(-ly, -ward)" -> "god", "godly", "godward"
        if '(-' in gloss:
            base = gloss.split('(')[0].strip()
            if base:
                cleaned.append(base)
            # Extract alternatives
            alt_match = re.search(r'\(([^)]+)\)', gloss)
            if alt_match:
                alts = alt_match.group(1)
                for alt in alts.split(','):
                    alt = alt.strip().lstrip('-')
                    if alt and len(alt) > 1:
                        # If starts with -, it's a suffix
                        if gloss.count('(-') > 0:
                            base_root = base.rstrip('- ')
                            cleaned.append(base_root + alt)
        else:
            cleaned.append(gloss)
    
    return cleaned


def build_lemma_to_glosses(strongs_dict: Dict[str, Dict]) -> Dict[str, Set[str]]:
    """
    Build mapping from Greek lemma to set of English glosses.
    """
    lemma_glosses = defaultdict(set)
    
    for strongs_id, data in strongs_dict.items():
        lemma = data['lemma']
        if not lemma:
            continue
        
        glosses = extract_glosses_from_kjv_def(data['kjv_def'])
        
        # Also extract key words from Strong's definition
        if data['strongs_def']:
            # Look for words after "to " (verb glosses)
            to_matches = re.findall(r'\bto\s+([a-z]+(?:\s+[a-z]+)?)', data['strongs_def'], re.IGNORECASE)
            for match in to_matches[:3]:  # Limit to first 3
                glosses.append(match.lower())
            
            # Look for "i.e. " clarifications
            ie_matches = re.findall(r'\bi\.e\.\s+([a-z]+(?:\s+[a-z]+)?)', data['strongs_def'], re.IGNORECASE)
            for match in ie_matches[:2]:
                glosses.append(match.lower())
        
        lemma_glosses[lemma].update(glosses)
    
    # Convert sets to lists for JSON serialization
    return {lemma: sorted(list(glosses)) for lemma, glosses in lemma_glosses.items()}


def build_lemma_to_strongs(strongs_dict: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Build mapping from Greek lemma to list of Strong's numbers.
    (Some lemmas may have multiple Strong's entries)
    """
    lemma_to_strongs = defaultdict(list)
    
    for strongs_id, data in strongs_dict.items():
        lemma = data['lemma']
        if lemma:
            lemma_to_strongs[lemma].append(strongs_id)
    
    return dict(lemma_to_strongs)


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    lexica_dir = base_dir / 'data' / 'raw' / 'lexica'
    output_dir = base_dir / 'data' / 'processed'
    
    # Parse Strong's dictionary
    strongs_xml_path = lexica_dir / 'dodson' / 'strongsgreek.xml'
    strongs_dict = parse_strongs_xml(strongs_xml_path)
    
    # Build mappings
    print("\nBuilding lemma-to-glosses mapping...")
    lemma_glosses = build_lemma_to_glosses(strongs_dict)
    print(f"  Created mappings for {len(lemma_glosses)} lemmas")
    
    print("\nBuilding lemma-to-Strong's mapping...")
    lemma_strongs = build_lemma_to_strongs(strongs_dict)
    print(f"  Created mappings for {len(lemma_strongs)} lemmas")
    
    # Save outputs
    print("\nSaving processed lexica...")
    
    # 1. Full Strong's dictionary
    strongs_output = output_dir / 'strongs_dictionary.json'
    with open(strongs_output, 'w', encoding='utf-8') as f:
        json.dump(strongs_dict, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {strongs_output}")
    
    # 2. Lemma to glosses (for dictionary priors in alignment)
    glosses_output = output_dir / 'lemma_to_glosses.json'
    with open(glosses_output, 'w', encoding='utf-8') as f:
        json.dump(lemma_glosses, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {glosses_output}")
    
    # 3. Lemma to Strong's numbers
    lemma_strongs_output = output_dir / 'lemma_to_strongs.json'
    with open(lemma_strongs_output, 'w', encoding='utf-8') as f:
        json.dump(lemma_strongs, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {lemma_strongs_output}")
    
    # Print some examples
    print("\n" + "="*60)
    print("Example entries:")
    print("="*60)
    
    example_lemmas = ['λόγος', 'θεός', 'ἀγάπη', 'πιστεύω']
    for lemma in example_lemmas:
        if lemma in lemma_glosses:
            strongs_nums = lemma_strongs.get(lemma, [])
            glosses = lemma_glosses[lemma]
            print(f"\n{lemma} ({', '.join(strongs_nums)})")
            print(f"  Glosses: {', '.join(glosses[:10])}")
    
    print("\n" + "="*60)
    print("✓ Lexica parsing complete!")
    print("="*60)


if __name__ == '__main__':
    main()

