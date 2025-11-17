# Word Alignment - Quick Start

Train a Greek-English word alignment model in 3 commands!

## 🚀 Run These Commands

```bash
# 1. Navigate to project
cd "/Users/arjun/Library/CloudStorage/GoogleDrive-arjungmenon@gmail.com/.shortcut-targets-by-id/1iKcqV2wXqpa76Q8hfY1h_V1N5NUzzgUr/UofT DL Term project/GreBerta-experiment-2"

# 2. Install dependencies (if not already done)
pip install -r requirements_alignment.txt

# 3. Train the alignment model!
python train_alignment.py --epochs 3 --batch-size 8

# That's it! Model will train for ~30-60 minutes on CPU
```

## ⚡ Quick Test (5 minutes)

Want to test the pipeline quickly first?

```bash
# Train on just 100 verses (5 minutes)
python train_alignment.py --quick-test --epochs 2
```

## ⏱️ Training Time

| Hardware | Time per Epoch | Total (3 epochs) |
|----------|----------------|------------------|
| CPU (M1 Mac) | ~45 minutes | ~2.5 hours |
| CPU (Intel) | ~60 minutes | ~3 hours |
| GPU (T4) | ~12 minutes | ~40 minutes |
| GPU (V100) | ~8 minutes | ~25 minutes |

**Tip:** Use `--quick-test` first to verify everything works, then run full training overnight!

## 📊 What to Expect

```
==================================================================================
TRAINING WORD ALIGNMENT MODEL
==================================================================================

Using device: cpu

1. Loading data...
  Train: 7,198 verses with alignments
  Dev:   284 verses
  Test:  443 verses

2. Loading tokenizers...
  ✓ Tokenizers loaded

3. Creating datasets...
  ✓ Created 900 training batches

4. Creating model...
  ✓ Model created
  Total parameters: 236,440,578
  Trainable parameters: 236,440,578

5. Setting up training...
  Epochs: 3
  Batch size: 8
  Learning rate: 2e-05
  Training steps: 2,700

==================================================================================
6. TRAINING STARTED
==================================================================================

Epoch 1/3
--------------------------------------------------------------------------------
Training: 100%|████████████████| 900/900 [45:23<00:00]
Train - Loss: 0.4523, P: 0.7234, R: 0.6891, F1: 0.7058
Dev   - P: 0.7423, R: 0.7012, F1: 0.7211
  ✓ Saved best model (F1: 0.7211)

[... training continues ...]

==================================================================================
✓ TRAINING COMPLETE!
==================================================================================

Model saved to: ./alignment_model_output
Test F1 Score: 0.7599

To use the model, see test_alignment.py
```

**Expected Results:**
- ✅ Training should complete without errors
- ✅ F1 score should be **0.70-0.85**
- ✅ Model saves to `./alignment_model_output/`

## 🧪 Test Your Model

After training completes:

```bash
# Test on a specific verse
python test_alignment.py --verse "Jude 1"

# Test on custom text
python test_alignment.py \
  --greek "Ἐν ἀρχῇ ἦν ὁ λόγος" \
  --english "In the beginning was the Word"

# Interactive mode - type your own text!
python test_alignment.py --interactive

# Show confidence scores
python test_alignment.py --verse "3 John 1" --show-confidence --threshold 0.4
```

## 📋 Example Output

```
==================================================================================
INTERLINEAR DISPLAY
==================================================================================
 1. Ἐν                    → In
 2. ἀρχῇ                  → beginning
 3. ἦν                    → was
 4. ὁ                     → the
 5. λόγος                 → Word
 6. καὶ                   → and
 7. ὁ                     → the
 8. λόγος                 → Word
 9. ἦν                    → was
10. πρὸς                  → with
11. τὸν                   → [no alignment]
12. θεόν                  → God
==================================================================================

✓ Found 11 alignments
```

## 🐛 Troubleshooting

### "Out of memory" Error

Reduce batch size:
```bash
python train_alignment.py --batch-size 4  # Or even 2
```

### "Model not found" when testing

You need to train first!
```bash
python train_alignment.py --quick-test  # Quick test
```

### Training is too slow

Options:
1. Use `--quick-test` for development (5 mins)
2. Train overnight on CPU
3. Use Google Colab free GPU: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Low F1 score (<0.60)

Try different hyperparameters:
```bash
# More conservative
python train_alignment.py --epochs 5 --batch-size 4 --lr 1e-5

# More aggressive
python train_alignment.py --epochs 5 --batch-size 16 --lr 5e-5
```

## 🎓 Understanding Metrics

**Precision**: Of predicted alignments, how many are correct?
- 0.75 = 75% of predictions are right ✅

**Recall**: Of true alignments, how many did we find?
- 0.75 = Found 75% of all alignments ✅

**F1 Score**: Balance of precision and recall
- 0.76 = Good overall performance! ✅

**What's a good F1?**
- **< 0.50**: Something's wrong 😞
- **0.50-0.65**: Okay (better than heuristics)
- **0.65-0.75**: Good! 😊
- **0.75-0.85**: Great! 🎉
- **> 0.85**: Excellent! 🌟

## 📈 Next Steps After Training

### 1. Generate Interlinear Bibles

```python
from test_alignment import AlignmentPredictor

predictor = AlignmentPredictor('./alignment_model_output')

# Load your verses
with open('my_verses.txt', 'r') as f:
    for line in f:
        greek, english = line.strip().split('\t')
        alignments = predictor.align(greek, english)
        # ... generate interlinear display
```

### 2. Analyze Model Performance

```bash
# Test on different verse types
python test_alignment.py --verse "Jude 1"        # Short epistle
python test_alignment.py --verse "3 John 1"      # Very short
python test_alignment.py --verse "Revelation 1:1" # Apocalyptic
```

### 3. Tune the Threshold

```bash
# High precision (fewer wrong alignments)
python test_alignment.py --verse "Jude 1" --threshold 0.7

# High recall (catch more alignments)
python test_alignment.py --verse "Jude 1" --threshold 0.3

# Balanced (default)
python test_alignment.py --verse "Jude 1" --threshold 0.5
```

### 4. Add More Data

After your model works on NT:
- Add Septuagint data (5x more training data)
- See `GETTING_STARTED.md` for Septuagint resources

## 🎯 Your Goal

Build an **interlinear Bible generator** that automatically aligns Greek and English text!

**You're 3 commands away from making it happen:** 🚀

```bash
pip install -r requirements_alignment.txt
python train_alignment.py --epochs 3
python test_alignment.py --verse "Jude 1"
```

## 📚 More Documentation

- **ALIGNMENT_GUIDE.md**: Complete technical guide
- **test_alignment.py**: Inference script with examples
- **train_alignment.py**: Training script (well-commented)

## 💡 Tips

1. **Start with `--quick-test`**: Verify everything works (5 mins)
2. **Monitor training**: F1 should improve each epoch
3. **Use test set carefully**: Only evaluate final model on test set
4. **Tune threshold**: Adjust based on your use case
5. **Compare versions**: Keep track of which hyperparameters work best

---

**Ready to train?** Copy the commands at the top and run them now! 🚀

