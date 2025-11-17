# NT Greek-English Alignment Scripts

This directory contains scripts for extracting and viewing word-level alignments between Greek (SBLGNT) and English (WEB) texts.

## Scripts

### `extract_alignments.py`

Extracts word-level alignment pairs from SBLGNT Greek NT and World English Bible using Strong's concordance numbers as weak supervision signals.

**Features:**
- Parses SBLGNT morphological Greek NT files
- Parses USFM English Bible files with embedded Strong's numbers
- Creates verse-level aligned parallel corpus
- Generates word-level alignment pairs using Strong's numbers
- Exports to JSON format with rich annotations

**Usage:**

```bash
# Extract alignments for one book
python extract_alignments.py --book 64-Jn-morphgnt.txt --output alignments_john.json

# Extract alignments for entire NT
python extract_alignments.py --output alignments_nt_full.json

# Specify custom data directory
python extract_alignments.py --data-dir /path/to/data --output output.json
```

**Output Format:**

The script generates a JSON file with:

```json
{
  "metadata": {
    "source_greek": "SBLGNT with MorphGNT annotations",
    "source_english": "World English Bible (WEB)",
    "alignment_method": "Strong's concordance numbers",
    "total_books": 27,
    "total_verses": 7925,
    "total_greek_tokens": 137536,
    "total_english_tokens": 216161,
    "total_alignments": 171085,
    "verses_with_alignments": 6510,
    "alignment_coverage": "82.1%"
  },
  "alignments": [
    {
      "book_code": "64",
      "book_name": "John",
      "chapter": 1,
      "verse": 1,
      "verse_id": "040101",
      "greek_tokens": [
        {
          "surface": "Ἐν",
          "word": "Ἐν",
          "normalized": "ἐν",
          "lemma": "ἐν",
          "pos": "P-",
          "parsing": "--------"
        },
        ...
      ],
      "english_tokens": [
        {
          "word": "In",
          "strongs": "G1722"
        },
        ...
      ],
      "alignments": [
        {
          "greek_idx": 0,
          "english_idx": 0,
          "greek_word": "Ἐν",
          "greek_lemma": "ἐν",
          "english_word": "In",
          "strongs": "G1722",
          "confidence": "strongs_match"
        },
        ...
      ]
    }
  ]
}
```

### `view_alignments.py`

View and inspect alignment pairs in human-readable format.

**Usage:**

```bash
# View specific verse
python view_alignments.py --verse "John 1:1"

# View first 5 verses
python view_alignments.py --limit 5

# View with visual alignment matrix
python view_alignments.py --verse "Romans 3:23" --visual

# Specify custom alignment file
python view_alignments.py --file ../data/processed/alignments_john.json --verse "John 3:16"
```

**Example Output:**

```
================================================================================
  John 1:1
================================================================================

Greek:   Ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν καὶ θεὸς ἦν ὁ λόγος
English: In the was the Word and the Word was with God and the Word was God beginning

Alignments (32 pairs):
--------------------------------------------------------------------------------
   1. Ἐν              (ἐν          ) → In              [G1722]
   2. Ἐν              (ἐν          ) → the             [G1722]
   ...
```

## Dataset Statistics

**Full NT Alignment Dataset:**
- **27 books** (Matthew through Revelation)
- **7,925 verses**
- **137,536 Greek tokens** (with POS, morphology, lemmas)
- **216,161 English tokens**
- **171,085 alignment pairs**
- **82.1% coverage** (6,510 verses with at least one alignment)
- **21.6 average alignments per verse**

## Data Quality Notes

### Alignment Method

Alignments are created using **Strong's concordance numbers** as weak supervision:

1. Each Greek lemma is mapped to Strong's number(s) via `lemma_to_strongs.json`
2. English words in WEB are tagged with Strong's numbers
3. When a Greek lemma's Strong's number matches an English word's Strong's number, an alignment is created

### Characteristics

**Strengths:**
- ✅ Large-scale automatic alignment (171k pairs)
- ✅ Based on scholarly lexicography (Strong's concordance)
- ✅ Includes full morphological annotations for Greek
- ✅ High coverage (82% of verses have alignments)

**Limitations:**
- ⚠️ **Noisy alignments**: Many-to-many mappings are common (e.g., Greek ἐν → English "in", "the", "with")
- ⚠️ **No word order information**: Strong's numbers don't capture positional alignment
- ⚠️ **Function words**: Articles, particles often over-aligned
- ⚠️ **Translational shifts**: Idiomatic translations not captured

### Use Cases

This dataset is suitable for:

1. **Training alignment models** - Provides weak supervision signals
2. **Cross-lingual representation learning** - Greek-English parallel data
3. **Contrastive learning** - Positive pairs (aligned) vs negative pairs
4. **Fine-tuning GreBerta** - Alignment as a downstream task
5. **Interlinear Bible generation** - Building word-level correspondences

## Next Steps

See the main project README for:
- How to use this data for fine-tuning GreBerta
- Creating train/dev/test splits
- Alignment model architectures
- Evaluation metrics

## Requirements

- Python 3.8+
- No external dependencies for basic usage
- Unicode NFC normalization for Greek text

## References

- **SBLGNT**: Society of Biblical Literature Greek New Testament
- **MorphGNT**: Morphological annotations by James Tauber
- **WEB**: World English Bible (Public Domain)
- **Strong's Concordance**: James Strong's Greek dictionary

