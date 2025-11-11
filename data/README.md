# NT Greek-English Alignment Data

## Step 1.1: Source Texts - COMPLETED ✓

### Greek Source: SBLGNT with MorphGNT Analysis
**Location:** `data/raw/greek/sblgnt/`

**Source:** MorphGNT SBLGNT Edition  
**License:** SBLGNT text (SBLGNT EULA), Morphological analysis (CC-BY-SA 3.0)  
**Citation:** Tauber, J. K., ed. (2017) _MorphGNT: SBLGNT Edition_. Version 6.12

**Format:** Tab-separated values with 7 columns:
1. Book/Chapter/Verse (e.g., `040101` = John 1:1)
2. Part of speech code (e.g., `N-` = noun, `V-` = verb, `P-` = preposition)
3. Parsing code (person, tense, voice, mood, case, number, gender)
4. Surface form with punctuation (e.g., `λόγος,`)
5. Word without punctuation (e.g., `λόγος`)
6. Normalized word (e.g., `λόγος`)
7. Lemma (e.g., `λόγος`)

**Example (John 1:1):**
```
040101 P- -------- Ἐν Ἐν ἐν ἐν
040101 N- ----DSF- ἀρχῇ ἀρχῇ ἀρχῇ ἀρχή
040101 V- 3IAI-S-- ἦν ἦν ἦν εἰμί
040101 RA ----NSM- ὁ ὁ ὁ ὁ
040101 N- ----NSM- λόγος, λόγος λόγος λόγος
```

**NT Books Available:** All 27 books (Matthew through Revelation)

---

### English Translation: World English Bible (WEB)
**Location:** `data/raw/english/`

**Source:** eBible.org  
**License:** Public Domain (CC0)  
**Format:** USFM (Unified Standard Format Markers)

**Key Features:**
- Literal translation suitable for alignment
- Includes Strong's numbers embedded in text
- Standard verse numbering matching Greek source
- USFM markup for structure

**Example (John 1:1):**
```
\v 1 \w In|strong="G1722"\w* \w the|strong="G1722"\w* beginning \w was|strong="G1510"\w* 
\w the|strong="G1722"\w* \w Word|strong="G3056"\w*, \w and|strong="G2532"\w* 
\w the|strong="G1722"\w* \w Word|strong="G3056"\w* \w was|strong="G1510"\w* 
\w with|strong="G1722"\w* \w God|strong="G2316"\w*, \w and|strong="G2532"\w* 
\w the|strong="G1722"\w* \w Word|strong="G3056"\w* \w was|strong="G1510"\w* 
\w God|strong="G2316"\w*.
```

**NT Books Available:** All 27 books (70-MATeng through 96-REVeng)

---

## Data Quality Notes

### Advantages of This Dataset:
1. **Rich Morphological Data:** Greek text includes lemmas, POS tags, and full morphological parsing
2. **Pre-aligned Strong's Numbers:** English WEB includes Strong's concordance numbers
3. **Verse-level Alignment:** Both texts use standard chapter:verse references
4. **Clean, Machine-Readable Formats:** Tab-separated (Greek) and structured markup (English)

---

## Step 1.2: Greek Linguistic Data & Lexica - COMPLETED ✓

### Lexica Downloaded

**Strong's Greek Dictionary (Dodson Edition)**
- **Location:** `data/raw/lexica/dodson/strongsgreek.xml` and `data/raw/lexica/strongs/`
- **Entries:** 5,523 Greek words
- **License:** Public Domain
- **Content:** Greek lemma, transliteration, KJV glosses, Strong's definitions

### Processed Lexicon Files

**1. Strong's Dictionary (`data/processed/strongs_dictionary.json`)**
```json
{
  "G3056": {
    "lemma": "λόγος",
    "translit": "lógos",
    "kjv_def": "account, cause, communication...",
    "strongs_def": "something said (including the thought)..."
  }
}
```

**2. Lemma to Glosses (`data/processed/lemma_to_glosses.json`)**
```json
{
  "λόγος": ["account", "cause", "communication", "doctrine", "word", ...],
  "θεός": ["god", "exceeding"],
  "ἀγάπη": ["love", "charity", "dear", "affection"]
}
```

**3. Lemma to Strong's Numbers (`data/processed/lemma_to_strongs.json`)**
```json
{
  "λόγος": ["G3056"],
  "θεός": ["G2316"]
}
```

### Key Statistics
- **5,516 Greek lemmas** mapped to English glosses
- **Average ~8 English glosses** per Greek lemma
- All lemmas normalized to Unicode NFC form

### Usage Notes
**Unicode Normalization:** Greek text uses NFC (Composed) form. Always normalize Greek strings with `unicodedata.normalize('NFC', text)` before lookups.

### Example Entries
| Greek | Strong's | English Glosses |
|-------|----------|-----------------|
| λόγος | G3056 | word, account, cause, doctrine, speech |
| θεός | G2316 | god, God |
| ἀγάπη | G26 | love, charity, affection |
| πιστεύω | G4100 | believe, commit, trust |
| ἄνθρωπος | G444 | man, human |

### Next Steps (Step 1.3):
- Parse SBLGNT Greek text files into structured format
- Parse USFM English text into clean token lists  
- Create unified verse-aligned parallel corpus in JSON format
- Verify MorphGNT data (lemmas, POS, morphology) for each Greek token

---

## Directory Structure

```
data/
├── raw/
│   ├── greek/
│   │   └── sblgnt/
│   │       ├── 61-Mt-morphgnt.txt  (Matthew)
│   │       ├── 62-Mk-morphgnt.txt  (Mark)
│   │       ├── 63-Lk-morphgnt.txt  (Luke)
│   │       ├── 64-Jn-morphgnt.txt  (John)
│   │       ├── 65-Ac-morphgnt.txt  (Acts)
│   │       ├── 66-Ro-morphgnt.txt  (Romans)
│   │       ├── ... (21 more NT books)
│   │       └── README.md
│   └── english/
│       ├── 70-MATeng-web.usfm  (Matthew)
│       ├── 71-MRKeng-web.usfm  (Mark)
│       ├── 72-LUKeng-web.usfm  (Luke)
│       ├── 73-JHNeng-web.usfm  (John)
│       ├── 74-ACTeng-web.usfm  (Acts)
│       ├── 75-ROMeng-web.usfm  (Romans)
│       └── ... (21 more NT books)
└── README.md (this file)
```

