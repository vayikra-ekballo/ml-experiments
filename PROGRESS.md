# NT Greek-English Alignment Project - Progress Log

## Phase 1: Foundation & Data Preparation

### ✅ Step 1.1: Acquire and Organize Source Texts (COMPLETED)

**What was done:**
- Downloaded SBLGNT Greek text (27 NT books)
- Downloaded World English Bible (WEB) translation
- Organized into `data/raw/greek/` and `data/raw/english/`
- Verified verse-level alignment

**Key findings:**
- SBLGNT includes MorphGNT annotations (lemmas, POS, morphology)
- WEB includes Strong's numbers embedded in text
- Both use standard verse numbering (easy alignment)

**Files created:**
- `data/raw/greek/sblgnt/` - 27 Greek NT books with morphology
- `data/raw/english/` - 27 English NT books in USFM format
- `data/README.md` - Documentation of data sources

---

### ✅ Step 1.2: Set Up Greek Linguistic Data & Lexica (COMPLETED)

**What was done:**
- Downloaded Strong's Greek Dictionary (Dodson edition)
- Parsed 5,523 Strong's entries from XML
- Created lemma→English glosses mapping (5,516 lemmas)
- Created lemma→Strong's number mapping
- Created Strong's→full entry dictionary

**Key findings:**
- Average ~8 English glosses per Greek lemma
- Unicode NFC normalization required for Greek text
- Glosses extracted from KJV definitions
- Additional glosses extracted from Strong's definitions

**Files created:**
- `data/raw/lexica/dodson/strongsgreek.xml` - Raw Strong's dictionary
- `data/raw/lexica/strongs/` - Additional Strong's resources
- `data/processed/strongs_dictionary.json` - Full Strong's data
- `data/processed/lemma_to_glosses.json` - Lemma→glosses mapping
- `data/processed/lemma_to_strongs.json` - Lemma→Strong's mapping
- `scripts/parse_lexica.py` - Lexicon parser
- `scripts/test_lexica.py` - Lexicon verification script

**Example data:**
```
λόγος (G3056): word, account, cause, doctrine, speech, matter...
θεός (G2316): god, God
ἀγάπη (G26): love, charity, affection
πιστεύω (G4100): believe, commit, trust
```

---

## Next Steps

### Step 1.3: Build Parallel Corpus (NEXT)

**Tasks:**
1. Parse SBLGNT Greek text files into structured format
   - Extract tokens, lemmas, POS, morphology, verse references
   - Handle punctuation and word boundaries
   
2. Parse USFM English text into clean token lists
   - Extract English words and verse references
   - Handle Strong's numbers embedded in text
   - Remove USFM markup

3. Create unified verse-aligned parallel corpus
   - Match Greek and English by book/chapter/verse
   - Output JSON format with:
     - Greek tokens with full linguistic features
     - English tokens with references
     - Verse-level metadata

4. Split into train/dev/test sets
   - Consider splitting by book or chapter
   - Maintain distribution of different writing styles

**Expected outputs:**
- `data/processed/parallel_corpus.json` - Full parallel data
- `data/processed/train.json` - Training set
- `data/processed/dev.json` - Development set
- `data/processed/test.json` - Test set

---

## Project Structure

```
nt-greek-align/
├── data/
│   ├── raw/
│   │   ├── greek/sblgnt/          # Greek NT with morphology
│   │   ├── english/               # English WEB translation
│   │   └── lexica/                # Greek-English dictionaries
│   ├── processed/                 # Processed/cleaned data
│   │   ├── strongs_dictionary.json
│   │   ├── lemma_to_glosses.json
│   │   └── lemma_to_strongs.json
│   └── README.md
├── scripts/
│   ├── parse_lexica.py           # Lexicon parser
│   └── test_lexica.py            # Lexicon tester
├── PROGRESS.md (this file)
└── README.md (to be created)
```

---

## Resources & Documentation

### Greek Text (SBLGNT)
- **Source:** https://github.com/morphgnt/sblgnt
- **License:** SBLGNT EULA (text), CC-BY-SA 3.0 (morphology)
- **Format:** Tab-separated: book/ch/vs | POS | parsing | surface | word | normalized | lemma

### English Text (WEB)
- **Source:** https://ebible.org
- **License:** Public Domain (CC0)
- **Format:** USFM with embedded Strong's numbers

### Lexica
- **Source:** https://github.com/morphgnt/strongs-dictionary-xml
- **License:** Public Domain
- **Entries:** 5,523 Greek words

### Important Notes
- **Unicode:** Always use NFC normalization for Greek text: `unicodedata.normalize('NFC', text)`
- **SBLGNT verse format:** `BBCCVV` where BB=book (01-27), CC=chapter, VV=verse
- **Strong's format:** `G####` where #### is 1-5624

---

## Timeline

- **Nov 11, 2024:** Step 1.1 completed
- **Nov 11, 2024:** Step 1.2 completed
- **Next:** Step 1.3 - Build parallel corpus

