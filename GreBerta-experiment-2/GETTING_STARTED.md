# Getting Started with GreBerta Fine-Tuning

## ✅ What We've Accomplished

### 1. Data Extraction Complete! 🎉

We've successfully extracted **171,085 Greek-English word alignment pairs** from the New Testament:

**Location:** `../nt-greek-align/data/processed/alignments_nt_full.json` (88 MB)

**Dataset Statistics:**
- ✅ 27 books (complete New Testament)
- ✅ 7,925 verses
- ✅ 137,536 Greek tokens (with full morphological annotations)
- ✅ 216,161 English tokens
- ✅ 171,085 alignment pairs
- ✅ 82.1% verse coverage

### 2. What the Data Contains

**For each verse, you have:**

**Greek tokens:**
- Surface form (e.g., "λόγος,")
- Normalized form (e.g., "λόγος")
- Lemma (e.g., "λόγος")
- POS tag (e.g., "N-" for noun)
- Full morphological parsing (e.g., "----NSM-" = nominative, singular, masculine)

**English tokens:**
- Word (e.g., "Word")
- Strong's number (e.g., "G3056")

**Alignment pairs:**
- Greek word index → English word index
- With Strong's number as linking key

### 3. How to View Your Data

```bash
cd ../nt-greek-align/scripts

# View John 1:1 (famous verse)
python view_alignments.py --verse "John 1:1"

# View Romans 3:23 (another famous verse)
python view_alignments.py --verse "Romans 3:23"

# View Matthew 5:3-5 (start of Sermon on the Mount)
python view_alignments.py --verse "Matthew 5:3"
python view_alignments.py --verse "Matthew 5:4"
python view_alignments.py --verse "Matthew 5:5"
```

## 🎯 Your Initial Question: Is This Sufficient?

### Question 1: Can we fine-tune for alignment tasks?

**Answer: YES! ✅**

The alignment task is **perfectly suited** for fine-tuning GreBerta, and you have excellent data for it:

1. **171k alignment pairs** - Large enough for supervised learning
2. **Strong's numbers provide weak supervision** - Noisy but useful
3. **Multiple alignment approaches possible**:
   - Token-pair classification (direct)
   - Contrastive learning (elegant)
   - Similarity-based (simple)

### Question 2: Is NT data sufficient?

**Answer: YES for initial fine-tuning! ✅**

**The NT dataset is sufficient because:**

| Factor | Status | Explanation |
|--------|--------|-------------|
| **Size** | ✅ Good | 138k Greek tokens, 7,925 verses |
| **Quality** | ✅ Excellent | Scholar-vetted SBLGNT with morphology |
| **Coverage** | ✅ Good | 82% of verses have alignments |
| **Task Type** | ✅ Fine-tuning | You're adapting, not pre-training from scratch |
| **Diversity** | ✅ Good | 27 books, multiple genres (narrative, epistles, apocalyptic) |

**Comparison to typical NLP datasets:**
- Stanford Sentiment: 11k sentences ← You have more!
- CoNLL NER: 20k sentences ← Similar size
- Penn Treebank: 40k sentences ← You're close
- Your NT: **7,925 verses** with **rich annotations**

**When to add more data:**
1. ✅ **Start with NT** - Validate approach, test pipeline
2. 📈 **Add Septuagint later** - If you want to improve performance (~600k more words)
3. 🚀 **Add other Koine texts** - For production-quality model

## 🚀 Quick Start: Three Paths Forward

### Path A: Start Simple → Build Up ⭐ RECOMMENDED

**Week 1: POS Tagging** (Simplest task)
- Fine-tune GreBerta for POS tagging (your data has tags for every word)
- Learn the fine-tuning pipeline
- Get comfortable with Hugging Face Trainer
- **Time**: 2-3 days
- **Code**: ~100 lines

**Week 2: Morphological Analysis**
- Predict case, number, gender, tense, etc.
- More complex than POS but still supervised token classification
- **Time**: 3-4 days
- **Code**: ~150 lines

**Week 3-4: Word Alignment**
- Now tackle the alignment task
- You'll have experience with the pipeline
- **Time**: 1 week
- **Code**: ~200-300 lines

### Path B: Go Straight to Alignment 🎯

If you're comfortable with deep learning:
- Jump directly to alignment task
- Use token-pair classification approach
- **Time**: 1-2 weeks
- **Code**: ~300 lines

### Path C: Research-Focused 📚

Compare multiple approaches:
1. Similarity-based alignment (baseline)
2. Token-pair classification
3. Contrastive learning
- **Time**: 3-4 weeks
- **Code**: ~500 lines

## 📝 Your Next Concrete Steps

### Step 1: Create Train/Dev/Test Splits (30 minutes)

```python
# Create: prepare_data.py
import json

with open('../nt-greek-align/data/processed/alignments_nt_full.json', 'r') as f:
    data = json.load(f)

# Split by books (better than random split)
train_books = ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
               '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
               'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
               '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James']
dev_books = ['1 Peter', '2 Peter', '1 John', '2 John']  
test_books = ['3 John', 'Jude', 'Revelation']

train_data = [v for v in data['alignments'] if v['book_name'] in train_books]
dev_data = [v for v in data['alignments'] if v['book_name'] in dev_books]
test_data = [v for v in data['alignments'] if v['book_name'] in test_books]

# Save splits
with open('train.json', 'w') as f:
    json.dump(train_data, f)
with open('dev.json', 'w') as f:
    json.dump(dev_data, f)
with open('test.json', 'w') as f:
    json.dump(test_data, f)

print(f"Train: {len(train_data)} verses")
print(f"Dev: {len(dev_data)} verses")
print(f"Test: {len(test_data)} verses")
```

### Step 2: Choose Your Starting Task

**Option 1: POS Tagging (EASIEST - recommended to start)**
- File to create: `train_pos_tagger.py`
- See example code in `README.md`

**Option 2: Word Alignment (YOUR GOAL)**
- File to create: `train_alignment.py`  
- More complex but directly addresses your goal

### Step 3: Set Up Environment

```bash
# If you want to reuse GreBerta-experiment-1 venv:
cd ../GreBerta-experiment-1
source .venv/bin/activate
pip install datasets evaluate scikit-learn

# Or create new venv:
cd ../GreBerta-experiment-2
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets evaluate scikit-learn
```

### Step 4: Verify GreBerta Loads

```bash
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModel

print("Loading GreBerta...")
tokenizer = AutoTokenizer.from_pretrained('bowphs/GreBerta')
model = AutoModel.from_pretrained('bowphs/GreBerta')

# Test on Greek text
text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
tokens = tokenizer.tokenize(text)
print(f"✓ Tokenized: {tokens}")
print(f"✓ Model loaded: {model.config.hidden_size} dimensions")
print("Ready to fine-tune!")
EOF
```

## 💡 Recommendations

### For Your Use Case (Alignment/Interlinear):

1. **Start with NT data only** ✅
   - It's sufficient for initial experiments
   - Faster iteration
   - Validates your approach

2. **Pick alignment as first task** if you're comfortable with DL ⭐
   - It's directly relevant to your goal
   - You have good data for it
   - More motivating than POS tagging

3. **OR start with POS tagging** if you want to learn the pipeline first 🎓
   - Much simpler
   - Same data, simpler task
   - Build confidence before tackling alignment

4. **Add Septuagint later** 📈
   - After you have working alignment model
   - If you want to improve performance
   - Not urgent for initial experiments

## 🎓 Learning Resources

### Alignment-Specific Papers:
1. **"Awesome Align"** (Dou & Neubig, 2021)
   - Uses BERT for word alignment
   - Very similar to what you want to do
   - [Paper](https://arxiv.org/abs/2101.08231)

2. **"SimAlign"** (Jalili Sabet et al., 2020)
   - Simple similarity-based alignment
   - Good baseline approach
   - [Paper](https://arxiv.org/abs/2004.08728)

### Tutorials:
- [Hugging Face Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [Training Custom Models](https://huggingface.co/docs/transformers/training)
- [GreBerta Model Card](https://huggingface.co/bowphs/GreBerta)

## ✅ Summary: You're Ready!

**What you have:**
- ✅ 171k alignment pairs from NT
- ✅ Full morphological annotations
- ✅ Scripts to view and explore data
- ✅ Clear documentation

**What to do next:**
1. **This week**: Create train/dev/test splits (30 min)
2. **This week**: Choose task (POS or alignment)
3. **Next week**: Implement training script
4. **Week 3**: Train, evaluate, iterate

**Your NT dataset is absolutely sufficient to:**
- Fine-tune GreBerta for alignment ✅
- Test different approaches ✅
- Build working interlinear demo ✅
- Potentially publish results ✅

**Questions?** 
- Check `README.md` for detailed technical info
- View example alignments with `view_alignments.py`
- Start with simple tasks if unsure (POS tagging)

## 🚀 Ready to Start?

Create your first script:

```bash
cd /path/to/GreBerta-experiment-2
touch prepare_data.py
# Copy the code from Step 1 above
python prepare_data.py
```

Then pick your task and start coding! 💪

