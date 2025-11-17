# Word Alignment Training Guide

Complete guide for training and using the Greek-English word alignment model.

## 🎯 What Is This?

This system learns to automatically align Greek words with their English translations, which is the foundation for building an interlinear Bible.

**Input:** Greek text + English translation  
**Output:** Which Greek words correspond to which English words

**Example:**
```
Greek:   Ἐν ἀρχῇ ἦν ὁ λόγος
English: In the beginning was the Word

Alignments:
  Ἐν    → In
  ἀρχῇ  → beginning
  ἦν    → was
  λόγος → Word
```

## 🏗️ Model Architecture

**Cross-Lingual Token-Pair Classification**

```
┌─────────────────┐         ┌─────────────────┐
│  Greek Text     │         │ English Text    │
│  "Ἐν ἀρχῇ ἦν"   │         │ "In beginning"  │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
  ┌─────────────┐           ┌─────────────┐
  │  GreBerta   │           │  BERT-base  │
  │   Encoder   │           │   Encoder   │
  └──────┬──────┘           └──────┬──────┘
         │                          │
         │  [embeddings]           [embeddings]
         │                          │
         └───────────┬──────────────┘
                     │
              [concatenate]
                     │
                     ▼
          ┌──────────────────┐
          │ Classification   │
          │     Head         │
          │  (3 layers)      │
          └────────┬─────────┘
                   │
                   ▼
            [Aligned / Not Aligned]
```

**Key Components:**
1. **GreBerta**: Encodes Greek text (pre-trained on Classical Greek)
2. **BERT**: Encodes English text
3. **Classifier**: Binary classification for each (Greek, English) pair
4. **Training**: Uses your 171k Strong's-based alignment pairs

## 📊 Dataset

**Training Data:**
- **143,886 alignment pairs** from 7,198 verses
- **Positive examples**: Greek-English pairs linked by Strong's numbers
- **Negative examples**: Random non-aligned pairs (2:1 ratio)

**Data Augmentation:**
- For each verse, we create:
  - All positive alignment pairs (from Strong's)
  - 2x negative pairs (random non-alignments)
- This balances the classes and improves learning

## 🚀 Quick Start

### Step 1: Train the Model

```bash
# Full training (~30-60 minutes on CPU, ~10-15 minutes on GPU)
python train_alignment.py --epochs 3 --batch-size 8

# Quick test on subset (~5 minutes)
python train_alignment.py --quick-test --epochs 2
```

**What to expect:**
- **Training time**: 30-60 minutes per epoch on CPU
- **Expected F1 score**: 0.70-0.85
- **Memory usage**: ~4-6GB RAM, ~2-4GB VRAM if GPU

### Step 2: Test the Model

```bash
# Test on specific verse
python test_alignment.py --verse "Jude 1"

# Test on custom text
python test_alignment.py \
  --greek "Ἐν ἀρχῇ ἦν ὁ λόγος" \
  --english "In the beginning was the Word"

# Interactive mode
python test_alignment.py --interactive

# Show confidence scores
python test_alignment.py --verse "3 John 1" --show-confidence
```

### Step 3: Generate Interlinear

```python
from test_alignment import AlignmentPredictor

# Load model
predictor = AlignmentPredictor('./alignment_model_output')

# Align text
greek = "καὶ ὁ λόγος σὰρξ ἐγένετο"
english = "and the Word became flesh"

alignments = predictor.align(greek, english, threshold=0.5)

# Print results
greek_words = greek.split()
english_words = english.split()

for g_idx, e_idx, conf in alignments:
    print(f"{greek_words[g_idx]} → {english_words[e_idx]} ({conf:.2f})")
```

## 📈 Training Parameters

### Default Settings (Recommended)

```python
--epochs 3           # Number of training passes
--batch-size 8       # Verses per batch (reduce if OOM)
--lr 2e-5           # Learning rate
--output-dir ./alignment_model_output
```

### For Faster Training

```python
--quick-test        # Train on 100 verses only (for testing)
--batch-size 16     # If you have GPU memory
```

### For Better Accuracy

```python
--epochs 5          # More training
--batch-size 4      # Smaller batches, more stable gradients
```

## 🎓 How Training Works

### 1. Data Preparation

For each verse, we create training examples:

```python
Verse: "Ἐν ἀρχῇ ἦν" / "In beginning was"

Positive pairs (from Strong's):
  (Ἐν, In) → 1 (aligned)
  (ἀρχῇ, beginning) → 1 (aligned)
  (ἦν, was) → 1 (aligned)

Negative pairs (random):
  (Ἐν, beginning) → 0 (not aligned)
  (ἀρχῇ, was) → 0 (not aligned)
  (ἦν, In) → 0 (not aligned)
  ... (2x as many negatives as positives)
```

### 2. Model Training

```
For each batch:
  1. Encode Greek with GreBerta
  2. Encode English with BERT
  3. For each (Greek, English) pair:
     - Get Greek embedding
     - Get English embedding
     - Concatenate
     - Classify: aligned or not
  4. Compute loss (cross-entropy)
  5. Update model weights
```

### 3. Evaluation

**Metrics:**
- **Precision**: Of predicted alignments, how many are correct?
- **Recall**: Of true alignments, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall

**Expected Performance:**
- **Baseline** (random): ~0.30 F1
- **Strong's heuristic**: ~0.65 F1
- **Our model**: ~0.70-0.85 F1 ✨

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution 1:** Reduce batch size
```bash
python train_alignment.py --batch-size 4
```

**Solution 2:** Use gradient accumulation (edit code)
```python
# In training loop, accumulate gradients
if step % 2 == 0:  # Update every 2 steps
    optimizer.step()
    optimizer.zero_grad()
```

**Solution 3:** Use CPU only
```bash
# Model will automatically use CPU if no GPU
```

### Issue: Training Too Slow

**Estimated times:**
- CPU (M1 Mac): ~45 minutes per epoch
- GPU (T4): ~12 minutes per epoch
- GPU (V100): ~8 minutes per epoch

**Solutions:**
1. Use `--quick-test` for development
2. Use Google Colab free GPU
3. Train overnight on CPU

### Issue: Model Not Loading

**Error:** `FileNotFoundError: alignment_model_output/model.pt`

**Solution:** Train the model first!
```bash
python train_alignment.py --quick-test  # Quick test
# Or
python train_alignment.py  # Full training
```

### Issue: Low F1 Score (<0.60)

**Possible causes:**
1. Learning rate too high → Try `--lr 1e-5`
2. Not enough epochs → Try `--epochs 5`
3. Batch size too large → Try `--batch-size 4`

**Solution:** Tune hyperparameters
```bash
# More conservative training
python train_alignment.py --epochs 5 --batch-size 4 --lr 1e-5
```

## 📊 Understanding Results

### Example Output

```
==================================================================================
Epoch 1/3
--------------------------------------------------------------------------------
Training: 100%|████████| 900/900 [45:23<00:00]
Train - Loss: 0.4523, P: 0.7234, R: 0.6891, F1: 0.7058
Dev   - P: 0.7423, R: 0.7012, F1: 0.7211
  ✓ Saved best model (F1: 0.7211)

Epoch 2/3
--------------------------------------------------------------------------------
Training: 100%|████████| 900/900 [44:12<00:00]
Train - Loss: 0.3214, P: 0.7891, R: 0.7523, F1: 0.7703
Dev   - P: 0.7723, R: 0.7334, F1: 0.7524
  ✓ Saved best model (F1: 0.7524)

Epoch 3/3
--------------------------------------------------------------------------------
Training: 100%|████████| 900/900 [43:58<00:00]
Train - Loss: 0.2567, P: 0.8123, R: 0.7823, F1: 0.7970
Dev   - P: 0.7834, R: 0.7512, F1: 0.7670
  ✓ Saved best model (F1: 0.7670)

==================================================================================
FINAL EVALUATION
==================================================================================

Dev set:
  Precision: 0.7834
  Recall:    0.7512
  F1 Score:  0.7670

Test set:
  Precision: 0.7712
  Recall:    0.7489
  F1 Score:  0.7599
```

**What this means:**
- **Precision 0.77**: When model says two words align, it's right 77% of the time
- **Recall 0.75**: Of all true alignments, model finds 75%
- **F1 0.76**: Overall quality score (pretty good!)

### Interpreting Predictions

```python
# Example prediction
alignments = [
    (0, 0, 0.95),  # Ἐν → In (95% confident) ✅ Very confident
    (1, 1, 0.72),  # ἀρχῇ → beginning (72% confident) ✅ Good
    (2, 2, 0.54),  # ἦν → was (54% confident) ⚠️ Uncertain
    (3, 3, 0.31),  # ὁ → the (31% confident) ❌ Below threshold
]

# Adjust threshold based on your needs:
threshold = 0.5  # Balanced (default)
threshold = 0.7  # High precision (fewer false positives)
threshold = 0.3  # High recall (catch more alignments)
```

## 🎨 Advanced Usage

### Custom Interlinear Formatter

```python
from test_alignment import AlignmentPredictor

predictor = AlignmentPredictor()

greek = "Ἐν ἀρχῇ ἦν ὁ λόγος"
english = "In the beginning was the Word"

alignments = predictor.align(greek, english)

# Create HTML interlinear
html = "<table>"
greek_words = greek.split()
english_words = english.split()

# Create alignment dict
align_dict = {}
for g_idx, e_idx, conf in alignments:
    if g_idx not in align_dict:
        align_dict[g_idx] = []
    align_dict[g_idx].append(english_words[e_idx])

# Generate HTML
for i, greek_word in enumerate(greek_words):
    english_str = " / ".join(align_dict.get(i, ["—"]))
    html += f"<tr><td>{greek_word}</td><td>{english_str}</td></tr>"

html += "</table>"
```

### Batch Alignment

```python
# Align multiple verses
verses = [
    ("Ἐν ἀρχῇ ἦν ὁ λόγος", "In the beginning was the Word"),
    ("καὶ ὁ λόγος ἦν πρὸς τὸν θεόν", "and the Word was with God"),
]

for greek, english in verses:
    alignments = predictor.align(greek, english)
    print(f"Found {len(alignments)} alignments")
```

## 📚 Next Steps

### After Training Works

1. **Analyze errors**: Which word types are hard to align?
2. **Tune threshold**: Balance precision vs recall
3. **Try different architectures**: Add attention layers
4. **Add more data**: Train on Septuagint (5x more data)

### Building Applications

1. **Web app**: Flask + HTML interlinear display
2. **Bible study tool**: Integrate with verse lookup
3. **Translation analyzer**: Compare alignment patterns
4. **Lexicon builder**: Extract word translation patterns

## 🔬 Model Details

**Architecture:**
- Greek encoder: GreBerta (125M params)
- English encoder: BERT-base (110M params)
- Classifier: 3-layer MLP (1M params)
- **Total: ~236M parameters**

**Training:**
- Optimizer: AdamW
- Learning rate: 2e-5 with linear warmup
- Batch size: 8 verses
- Gradient clipping: 1.0
- Training time: ~2-3 hours on CPU

**Data:**
- 7,198 training verses
- ~20 alignment pairs per verse (avg)
- 2:1 negative:positive ratio
- Total: ~430k training pairs

## 📖 References

### Related Work

1. **Awesome Align** (Dou & Neubig, 2021)
   - Uses multilingual BERT for alignment
   - State-of-the-art on parallel corpus alignment
   - https://arxiv.org/abs/2101.08231

2. **SimAlign** (Jalili Sabet et al., 2020)
   - Similarity-based alignment
   - No training required (good baseline)
   - https://arxiv.org/abs/2004.08728

3. **GreBerta** (Riemenschneider & Frank, 2023)
   - Pre-trained model for Ancient Greek
   - https://arxiv.org/abs/2305.13698

### Our Approach

We combine:
- GreBerta's Ancient Greek knowledge
- Token-pair classification (like Awesome Align)
- Strong's numbers for weak supervision

This gives us a model specifically tuned for Greek-English Bible alignment!

## ✅ Checklist

Before training:
- [ ] Data splits created (`data/train.json` exists)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Enough disk space (~2GB for model checkpoints)

After training:
- [ ] Model saved to `alignment_model_output/`
- [ ] Test F1 score > 0.65
- [ ] Tested on sample verses
- [ ] Interlinear generation works

Ready for production:
- [ ] Tested on full test set
- [ ] Error analysis completed
- [ ] Threshold tuned for use case
- [ ] Documentation updated

## 🎓 Tips for Success

1. **Start small**: Use `--quick-test` to verify pipeline
2. **Monitor metrics**: F1 should improve each epoch
3. **Save checkpoints**: Training can take hours
4. **Tune threshold**: Adjust based on precision/recall needs
5. **Compare with baseline**: Random alignment ~0.30 F1

**Your goal:** Beat Strong's heuristic (0.65 F1) with learned model!

Good luck! 🚀

