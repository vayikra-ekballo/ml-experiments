# GreBerta Fine-Tuning Project - Complete Index

Everything you need for Greek-English NLP with GreBerta.

## 📚 Start Here

**New to this project?**
→ Read `GETTING_STARTED.md`

**Want to train POS tagger?**
→ Read `QUICKSTART.md`

**Want to train word alignment?**
→ Read `ALIGNMENT_QUICKSTART.md`

**Need technical details?**
→ Read `README.md` or `ALIGNMENT_GUIDE.md`

## 📂 Project Structure

```
GreBerta-experiment-2/
│
├── 📖 DOCUMENTATION
│   ├── INDEX.md                        ← You are here
│   ├── GETTING_STARTED.md              ← Beginner guide
│   ├── README.md                       ← Technical details for all tasks
│   ├── SUMMARY.md                      ← Project overview
│   │
│   ├── QUICKSTART.md                   ← POS tagging quick start
│   ├── ALIGNMENT_QUICKSTART.md         ← Alignment quick start
│   ├── ALIGNMENT_GUIDE.md              ← Alignment technical guide
│   └── ALIGNMENT_COMPLETE.md           ← Alignment system overview
│
├── 🔧 SCRIPTS
│   ├── prepare_data.py                 ← Create train/dev/test splits
│   │
│   ├── train_pos_tagger.py             ← Train POS tagger
│   ├── test_pos_model.py               ← Test POS tagger
│   │
│   ├── train_alignment.py              ← Train word alignment
│   ├── test_alignment.py               ← Test word alignment
│   │
│   └── greberta_fine_tune.py           ← Template for custom tasks
│
├── 📊 DATA
│   └── data/
│       ├── train.json                  ← 7,198 training verses
│       ├── dev.json                    ← 284 dev verses
│       └── test.json                   ← 443 test verses
│
├── 📦 DEPENDENCIES
│   ├── requirements.txt                ← General requirements
│   └── requirements_alignment.txt      ← Alignment-specific
│
└── 💾 OUTPUTS (created after training)
    ├── pos_tagger_output/              ← POS tagger model
    └── alignment_model_output/         ← Alignment model
```

## 🎯 Quick Links by Task

### Task 1: POS Tagging (Easier)

**Purpose:** Tag Greek words with parts of speech (Noun, Verb, etc.)

**Files you need:**
- 📖 `QUICKSTART.md` - How to train
- 🔧 `train_pos_tagger.py` - Training script
- 🔧 `test_pos_model.py` - Testing script

**Commands:**
```bash
pip install -r requirements.txt
python train_pos_tagger.py --epochs 3
python test_pos_model.py
```

**Time:** ~20 minutes on CPU

---

### Task 2: Word Alignment (Your Goal)

**Purpose:** Align Greek words with English translations (for interlinear Bibles)

**Files you need:**
- 📖 `ALIGNMENT_QUICKSTART.md` - Quick start
- 📖 `ALIGNMENT_GUIDE.md` - Technical guide
- 📖 `ALIGNMENT_COMPLETE.md` - System overview
- 🔧 `train_alignment.py` - Training script
- 🔧 `test_alignment.py` - Testing script

**Commands:**
```bash
pip install -r requirements_alignment.txt
python train_alignment.py --epochs 3
python test_alignment.py --verse "Jude 1"
```

**Time:** ~2-3 hours on CPU, ~40 mins on GPU

---

## 📖 Documentation Guide

### For Beginners

1. **Start:** `GETTING_STARTED.md`
   - Project overview
   - Data description
   - Next steps guide

2. **Choose task:**
   - POS tagging → `QUICKSTART.md`
   - Word alignment → `ALIGNMENT_QUICKSTART.md`

3. **Run training** following the quick start guide

4. **Test your model** with the test scripts

### For Technical Details

1. **`README.md`**
   - All task architectures
   - Code examples
   - Hyperparameter options

2. **`ALIGNMENT_GUIDE.md`**
   - Alignment architecture
   - Training details
   - Advanced usage
   - Troubleshooting

3. **`ALIGNMENT_COMPLETE.md`**
   - System overview
   - Use cases
   - Performance metrics
   - Next steps

### For Reference

1. **`SUMMARY.md`**
   - Project status
   - What's been accomplished
   - File manifest

2. **`INDEX.md`** (this file)
   - Navigation guide
   - Quick links
   - File descriptions

## 🚀 Common Workflows

### Workflow 1: Quick Start (POS Tagging)

```bash
# 1. Read the guide (5 mins)
open QUICKSTART.md

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train (20 mins)
python train_pos_tagger.py --epochs 3

# 4. Test
python test_pos_model.py
```

**Total time:** ~30 minutes

---

### Workflow 2: Word Alignment Training

```bash
# 1. Read the guide (10 mins)
open ALIGNMENT_QUICKSTART.md

# 2. Install dependencies
pip install -r requirements_alignment.txt

# 3. Quick test first (5 mins)
python train_alignment.py --quick-test --epochs 2

# 4. Full training (2-3 hours)
python train_alignment.py --epochs 3 --batch-size 8

# 5. Test and generate interlinear
python test_alignment.py --verse "Jude 1"
python test_alignment.py --interactive
```

**Total time:** ~3-4 hours (mostly unattended training)

---

### Workflow 3: Custom Task Development

```bash
# 1. Read architecture guide
open README.md

# 2. Start from template
cp greberta_fine_tune.py my_custom_task.py

# 3. Modify for your task
# (See README.md for examples)

# 4. Train
python my_custom_task.py
```

---

## 📊 Data Overview

### What You Have

**Source:** New Testament (SBLGNT + World English Bible)

**Statistics:**
- **27 books** (Matthew through Revelation)
- **7,925 verses** total
- **137,536 Greek tokens** with morphology
- **171,085 alignment pairs**

**Splits:**
```
Train:  7,198 verses (91%)
Dev:      284 verses (3.6%)
Test:     443 verses (5.6%)
```

### Data Format

Each verse contains:
- Greek tokens (word, lemma, POS, morphology)
- English tokens (word, Strong's number)
- Alignment pairs (Greek index → English index)

See `GETTING_STARTED.md` for detailed format description.

---

## 🎓 Learning Path

### Beginner Path (1 week)

**Day 1-2:**
- [ ] Read `GETTING_STARTED.md`
- [ ] Understand data structure
- [ ] Run `prepare_data.py`

**Day 3-4:**
- [ ] Train POS tagger (`QUICKSTART.md`)
- [ ] Test and analyze results
- [ ] Understand metrics

**Day 5-7:**
- [ ] Read `ALIGNMENT_QUICKSTART.md`
- [ ] Train alignment model
- [ ] Generate interlinear displays

### Advanced Path (2-3 weeks)

**Week 1:**
- [ ] Train both POS and alignment
- [ ] Compare performance
- [ ] Tune hyperparameters

**Week 2:**
- [ ] Error analysis
- [ ] Try different architectures
- [ ] Implement improvements

**Week 3:**
- [ ] Build application (interlinear generator)
- [ ] Process full Bible
- [ ] Document results

### Research Path (1-2 months)

**Phase 1:**
- [ ] Establish baselines
- [ ] Train multiple variants
- [ ] Compare with state-of-the-art

**Phase 2:**
- [ ] Add Septuagint data
- [ ] Implement advanced architectures
- [ ] Cross-validate results

**Phase 3:**
- [ ] Write paper/blog post
- [ ] Release models
- [ ] Build public demo

---

## ⚡ Quick Commands Reference

### Training

```bash
# POS tagging
python train_pos_tagger.py --epochs 3

# Word alignment (quick test)
python train_alignment.py --quick-test

# Word alignment (full)
python train_alignment.py --epochs 3 --batch-size 8
```

### Testing

```bash
# POS tagger
python test_pos_model.py

# Alignment (specific verse)
python test_alignment.py --verse "Jude 1"

# Alignment (interactive)
python test_alignment.py --interactive

# Alignment (custom text)
python test_alignment.py \
  --greek "Ἐν ἀρχῇ ἦν ὁ λόγος" \
  --english "In the beginning was the Word"
```

### Data Preparation

```bash
# Create splits (already done)
python prepare_data.py

# View alignment data
cd ../nt-greek-align/scripts
python view_alignments.py --verse "John 1:1"
```

---

## 🆘 Getting Help

### For Training Issues

1. Check `ALIGNMENT_GUIDE.md` troubleshooting section
2. Verify dependencies installed: `pip list`
3. Check data exists: `ls data/`
4. Try quick-test first: `--quick-test`

### For Code Questions

1. Read docstrings in training scripts
2. Check `README.md` for examples
3. Look at `ALIGNMENT_GUIDE.md` architecture section

### For Results Questions

1. Compare with expected metrics in guides
2. Check `ALIGNMENT_COMPLETE.md` performance section
3. Try tuning hyperparameters

---

## 🎯 Your Next Action

**Choose your path:**

### Option A: POS Tagging (Easier, 30 minutes)
```bash
open QUICKSTART.md  # Read this first
python train_pos_tagger.py --epochs 3
```

### Option B: Word Alignment (Your Goal, 3 hours)
```bash
open ALIGNMENT_QUICKSTART.md  # Read this first
python train_alignment.py --epochs 3
```

### Option C: Learn More First
```bash
open GETTING_STARTED.md  # Comprehensive intro
open ALIGNMENT_COMPLETE.md  # System overview
```

---

## ✅ Final Checklist

Before you start:
- [ ] Read appropriate quick start guide
- [ ] Dependencies installed
- [ ] Data splits created
- [ ] Enough disk space (~2GB)

After training:
- [ ] Model saved successfully
- [ ] Metrics look good (F1 > 0.65)
- [ ] Tested on sample data
- [ ] Ready for application!

---

**Choose a quick start guide above and begin!** 🚀

Everything is ready for you to start training. Good luck!

