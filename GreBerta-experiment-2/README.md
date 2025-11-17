# GreBerta Fine-Tuning for Koine Greek

This project aims to fine-tune the GreBerta model (trained on Classical Greek) for Koine Greek NLP tasks, particularly word alignment for interlinear Bible generation.

## 🎯 Project Goals

1. **Domain Adaptation**: Adapt GreBerta from Classical Greek → Koine Greek
2. **Word Alignment**: Train model to align Greek words with English translations
3. **Interlinear Generation**: Build an automatic Greek-English interlinear system

## 📊 Available Data

### Complete NT Alignment Dataset ✅

**Location:** `../nt-greek-align/data/processed/alignments_nt_full.json`

**Statistics:**
- **27 books** (entire New Testament)
- **7,925 verses**
- **137,536 Greek tokens** (with POS, morphology, lemmas)
- **216,161 English tokens**
- **171,085 alignment pairs** (Greek ↔ English)
- **82.1% coverage** of verses

**Data Quality:**
- Alignments extracted using Strong's concordance numbers (weak supervision)
- Each Greek token has: surface form, lemma, POS tag, full morphological parsing
- Each English token has: word, Strong's number (where available)

### How to View the Data

```bash
cd ../nt-greek-align/scripts

# View John 1:1
python view_alignments.py --verse "John 1:1"

# View Romans 3:23
python view_alignments.py --verse "Romans 3:23"

# View first 5 verses
python view_alignments.py --limit 5
```

## 🚀 Next Steps for Fine-Tuning

### Step 1: Data Preparation

**Create train/dev/test splits:**

```python
import json
import random
from pathlib import Path

# Load full dataset
with open('../nt-greek-align/data/processed/alignments_nt_full.json', 'r') as f:
    data = json.load(f)

# Split by books for better distribution
# Suggestion: 
#   - Train: 20 books (~80%)
#   - Dev: 4 books (~15%)
#   - Test: 3 books (~5%)

train_books = ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', 
               '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
               'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
               '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James']
dev_books = ['1 Peter', '2 Peter', '1 John', '2 John']
test_books = ['3 John', 'Jude', 'Revelation']

# Filter alignments by book
train_data = [v for v in data['alignments'] if v['book_name'] in train_books]
dev_data = [v for v in data['alignments'] if v['book_name'] in dev_books]
test_data = [v for v in data['alignments'] if v['book_name'] in test_books]

# Save splits
# ... save to files ...
```

### Step 2: Choose Fine-Tuning Approach

#### Option A: Alignment as Token-Pair Classification ⭐ Recommended

**Task**: For each (Greek word, English word) pair, predict if they're aligned

**Architecture**:
```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class AlignmentModel(nn.Module):
    def __init__(self, greek_model='bowphs/GreBerta', english_model='bert-base-uncased'):
        super().__init__()
        self.greek_encoder = AutoModel.from_pretrained(greek_model)
        self.english_encoder = AutoModel.from_pretrained(english_model)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),  # Concatenate Greek + English embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary: aligned or not
        )
    
    def forward(self, greek_input_ids, greek_attention_mask,
                english_input_ids, english_attention_mask):
        # Encode Greek
        greek_out = self.greek_encoder(greek_input_ids, greek_attention_mask)
        greek_embeddings = greek_out.last_hidden_state
        
        # Encode English
        english_out = self.english_encoder(english_input_ids, english_attention_mask)
        english_embeddings = english_out.last_hidden_state
        
        # For each (Greek, English) pair, concatenate and classify
        # ... implementation details ...
        
        return logits
```

**Pros:**
- Directly matches your goal (word alignment)
- Uses your Strong's-based alignments as training labels
- Fine-tunes GreBerta for Koine Greek during alignment task

**Cons:**
- Requires cross-lingual architecture (Greek + English encoders)
- Many negative examples (non-aligned pairs)

#### Option B: Contrastive Learning for Alignment

**Task**: Learn embeddings where aligned words are close in vector space

**Architecture**: Use contrastive loss (e.g., InfoNCE)
- Positive pairs: aligned Greek-English words
- Negative pairs: non-aligned words from same verse

**Pros:**
- Elegant approach for learning cross-lingual representations
- Handles many-to-many alignments naturally

**Cons:**
- Requires careful negative sampling
- Indirect approach (embeddings → alignment via similarity)

#### Option C: Start with Simpler Tasks First 🎯 Good for Learning

Before tackling alignment, warm up with simpler supervised tasks:

1. **POS Tagging** (easiest):
   - Your data has POS tags for every Greek word
   - Standard token classification task
   - ~100 lines of code with Hugging Face Trainer

2. **Lemmatization**:
   - Predict lemma for each Greek word
   - Also a token classification task (or seq2seq)

3. **Morphological Analysis**:
   - Predict case, number, gender, tense, etc.
   - Multi-task token classification

**Why start here?**
- Test your pipeline on simpler tasks
- Verify GreBerta loads and trains correctly
- Understand the training loop before tackling alignment
- These tasks also adapt GreBerta to Koine Greek

### Step 3: Implementation Template

**For POS Tagging (simplest starting point):**

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# 1. Load model
model_name = 'bowphs/GreBerta'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get unique POS tags from your data
pos_tags = set()
for verse in alignments:
    for token in verse['greek_tokens']:
        pos_tags.add(token['pos'])
pos_to_id = {tag: i for i, tag in enumerate(sorted(pos_tags))}

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(pos_to_id)
)

# 2. Prepare dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples['pos_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore padding
            else:
                label_ids.append(pos_to_id[label[word_id]])
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Convert your data to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    'tokens': [verse['greek_tokens'] for verse in train_data],
    'pos_tags': [[t['pos'] for t in verse['greek_tokens']] for verse in train_data]
})

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)

# 3. Train
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()
```

### Step 4: Evaluation Metrics

**For Word Alignment:**
- **Precision**: % of predicted alignments that are correct
- **Recall**: % of gold alignments that were predicted
- **F1 Score**: Harmonic mean of precision and recall
- **AER** (Alignment Error Rate): Standard metric in MT alignment

**For POS Tagging:**
- **Accuracy**: % of tokens with correct POS tag
- **Per-class F1**: F1 for each POS category

## 📚 Recommended Learning Path

### Week 1: Foundation
1. ✅ Understand GreBerta architecture (completed in experiment-1)
2. ✅ Extract alignment data (completed!)
3. Create train/dev/test splits
4. Write data loading code

### Week 2: Simple Fine-Tuning
1. Implement POS tagging fine-tuning
2. Train and evaluate on dev set
3. Analyze errors, iterate
4. Document results

### Week 3: Alignment Model
1. Design alignment architecture
2. Implement training loop
3. Train on alignment data
4. Evaluate alignment quality

### Week 4: Application & Analysis
1. Build interlinear generation demo
2. Error analysis
3. Compare with baseline methods
4. Write up results

## 🎓 Additional Resources

### Alignment Papers
- "Awesome Align" (Dou & Neubig, 2021) - Uses BERT for alignment
- "SimAlign" (Jalili Sabet et al., 2020) - Similarity-based alignment

### Hugging Face Resources
- [Token Classification Tutorial](https://huggingface.co/docs/transformers/tasks/token_classification)
- [Custom Training Loop](https://huggingface.co/docs/transformers/training)
- [Datasets Documentation](https://huggingface.co/docs/datasets/)

### Greek NLP
- GreBerta Paper: "Exploring Large Language Models for Classical Philology"
- Universal Dependencies Ancient Greek Treebank

## 🛠️ Setup

```bash
# Install dependencies
pip install torch transformers datasets evaluate scikit-learn

# Verify GreBerta loads
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bowphs/GreBerta'); print('✓')"
```

## 📝 Questions to Consider

1. **Is NT data sufficient?**
   - Yes for initial experiments (7,925 verses)
   - Consider adding Septuagint later (~30,000 more verses)

2. **Which task should I start with?**
   - Start with POS tagging (simplest, ~2 days to implement)
   - Then move to alignment (more complex, ~1 week)

3. **Do I need GPU?**
   - Recommended but not required
   - CPU training: slower but feasible for NT-sized data
   - Use Google Colab free GPU if needed

## 🎯 Success Criteria

**Minimum Viable Product:**
- ✅ Fine-tuned GreBerta on Koine Greek data
- ✅ Alignment model with >70% F1 score
- ✅ Demo interlinear for a few verses

**Stretch Goals:**
- 📈 Add Septuagint data
- 📊 Compare multiple alignment approaches
- 🚀 Deploy as web demo
- 📄 Write paper/blog post

## Next File to Create

Start with: `prepare_data.py` - Script to create train/dev/test splits

