# Quick Start: Train POS Tagger NOW

## 🚀 Run These Commands

```bash
# 1. Navigate to project
cd "/Users/arjun/Library/CloudStorage/GoogleDrive-arjungmenon@gmail.com/.shortcut-targets-by-id/1iKcqV2wXqpa76Q8hfY1h_V1N5NUzzgUr/UofT DL Term project/GreBerta-experiment-2"

# 2. Set up environment (reuse from experiment-1 or create new)
# Option A: Reuse existing
cd ../GreBerta-experiment-1 && source .venv/bin/activate && cd ../GreBerta-experiment-2

# Option B: Create new
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify data exists
ls -lh data/*.json

# 5. Train POS tagger (3 epochs, ~10-20 minutes on CPU, 2-5 minutes on GPU)
python train_pos_tagger.py --epochs 3 --batch-size 16 --lr 2e-5

# That's it! The model will train and save automatically.
```

## ⏱️ Expected Training Time

- **CPU (M1/M2 Mac)**: ~10-20 minutes
- **GPU (if available)**: ~2-5 minutes
- **Google Colab (free GPU)**: ~3-5 minutes

## 📊 What Will Happen

1. Loads your 7,198 training verses
2. Tokenizes Greek text with GreBerta tokenizer
3. Fine-tunes for POS tagging (13 labels)
4. Evaluates on dev set after each epoch
5. Saves best model to `./pos_tagger_output/`
6. Reports final test accuracy

**Expected Accuracy**: ~95-98% (POS tagging is relatively easy)

## 🎯 After Training

Test your model:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForTokenClassification.from_pretrained('./pos_tagger_output')
tokenizer = AutoTokenizer.from_pretrained('./pos_tagger_output')

# Test on Greek text
text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get predictions
predictions = outputs.logits.argmax(-1)[0]
tokens = tokenizer.tokenize(text)

# Show results
for token, pred in zip(tokens, predictions):
    label = model.config.id2label[pred.item()]
    print(f"{token:15s} -> {label}")
```

## 🐛 Troubleshooting

**Error: `No module named 'transformers'`**
```bash
pip install -r requirements.txt
```

**Error: `FileNotFoundError: data/train.json`**
```bash
python prepare_data.py
```

**Out of memory error:**
```bash
# Reduce batch size
python train_pos_tagger.py --batch-size 8
```

## 📝 Next Steps After POS Tagging Works

1. ✅ Analyze errors (what POS tags are confused?)
2. ✅ Try different hyperparameters
3. 🎯 Move to word alignment task (your main goal)

See `README.md` for alignment implementation details.

