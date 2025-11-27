# Improved Word Alignment Model (V2)

This document describes the significant improvements made to the Greek-English word alignment model in version 2.

## Key Improvements Over V1

### 1. Architecture Improvements

#### Cross-Attention Between Languages
- Added bidirectional multi-head attention between Greek and English representations
- Greek embeddings attend to English context and vice versa
- This helps the model understand how words relate across languages

```python
# Greek attends to English
greek_ctx = cross_attention(greek_proj, english_proj, english_proj)
greek_proj = greek_proj + greek_ctx  # Residual connection
```

#### Richer Feature Representation
- **Cosine similarity**: Explicit similarity measure between word pairs
- **Bilinear interaction**: Learnable interaction between Greek and English embeddings
- **Positional encoding**: Captures relative position differences between aligned words

#### Proper Subword Pooling
- V1 used only the first subword token for each word
- V2 averages all subword tokens to get word-level embeddings
- This is especially important for morphologically rich Greek words

### 2. Training Improvements

#### Balanced Data Sampling
- V1: 2:1 negative:positive ratio (biased toward predicting "not aligned")
- V2: 1:1 balanced ratio for better learning

#### Hard Negative Mining
- V1: Random negative sampling (too easy for the model)
- V2: Strategic sampling of challenging negatives:
  - Positionally close words that aren't aligned
  - Words with similar context
  - Words that share alignment partners with other words

#### Loss Function Options
- **Weighted Cross-Entropy**: Slight upweight (1.5x) for positive class
- **Focal Loss**: Down-weights easy examples, focuses on hard cases
- **Contrastive Loss**: InfoNCE-style loss that pulls aligned pairs together

#### Threshold Optimization
- Automatically finds the optimal classification threshold on the dev set
- No more guessing - the model tells you the best threshold to use

### 3. Inference Improvements

#### Multiple Alignment Strategies
- `greedy`: Take all pairs above threshold
- `best_match`: For each Greek word, take the best English match
- `all`: Return all pairs with probabilities for custom processing

#### Coverage Post-Processing
- Ensures content words (nouns, verbs, etc.) have at least one alignment
- Uses a lower threshold for unaligned content words
- Prevents "holes" in the interlinear display

#### Greek Function Word Detection
- Identifies articles, particles, prepositions that may not have direct alignments
- Prevents false positives from over-aligning function words

## Usage

### Training

```bash
# Basic training (recommended first)
python train_alignment_v2.py --epochs 5 --batch-size 8

# With contrastive learning (best quality)
python train_alignment_v2.py --epochs 5 --use-contrastive

# Quick test run
python train_alignment_v2.py --quick-test

# All options
python train_alignment_v2.py \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --use-contrastive \
    --contrastive-weight 0.5 \
    --use-focal-loss \
    --freeze-epochs 1 \
    --output-dir ./alignment_model_v2_output
```

### Testing

```bash
# Test on examples
python test_alignment_v2.py

# Custom text
python test_alignment_v2.py --greek "Ἐν ἀρχῇ ἦν ὁ λόγος" --english "In the beginning was the Word"

# Interactive mode
python test_alignment_v2.py --interactive

# Different strategies
python test_alignment_v2.py --strategy best_match
python test_alignment_v2.py --strategy greedy --threshold 0.4
```

### Compare V1 vs V2

```bash
python compare_models.py
```

## Expected Results

With proper training, V2 should achieve:
- **F1 Score**: 0.85+ (vs ~0.70 for V1)
- **Better recall**: Fewer missed alignments
- **Higher precision**: Fewer false alignments

For "Ἐν ἀρχῇ ἦν ὁ λόγος" → "In the beginning was the Word":

| Greek | V1 | V2 (Expected) |
|-------|-----|---------------|
| Ἐν | In, the, the | In |
| ἀρχῇ | [no alignment] | beginning |
| ἦν | was | was |
| ὁ | was | the |
| λόγος | Word, beginning | Word |

## Model Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   GreBerta      │     │     BERT        │
│  (Greek enc.)   │     │  (English enc.) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    ┌────────┐              ┌────────┐
    │ Project │              │ Project │
    │ to 384  │              │ to 384  │
    └────┬───┘              └────┬───┘
         │                       │
         ▼                       ▼
    ┌───────────────────────────────────┐
    │     Cross-Attention (optional)     │
    │  Greek ←→ English contextualize   │
    └────────────────┬──────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────┴────┐             ┌────┴────┐
    │  Word   │             │  Word   │
    │ Pooling │             │ Pooling │
    └────┬────┘             └────┬────┘
         │                       │
         └───────┬───────────────┘
                 │
    ┌────────────┴────────────┐
    │    Pair Features        │
    │  - Concatenation        │
    │  - Bilinear interaction │
    │  - Cosine similarity    │
    │  - Position encoding    │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │      Classifier         │
    │   (3 layers + LayerNorm)│
    └────────────┬────────────┘
                 │
                 ▼
         [Aligned / Not Aligned]
```

## Files

- `train_alignment_v2.py` - Training script with all improvements
- `test_alignment_v2.py` - Inference with post-processing
- `compare_models.py` - Side-by-side comparison tool
- `requirements_alignment.txt` - Dependencies

## Troubleshooting

### "V2 model appears worse"
- Training may need more epochs (try 8-10)
- Try enabling contrastive learning: `--use-contrastive`
- Check if training data was properly loaded

### Out of memory
- Reduce batch size: `--batch-size 4`
- Enable gradient checkpointing (add to model if needed)

### Training is slow
- V2 is more compute-intensive due to cross-attention
- Use GPU if available
- Consider `--freeze-epochs 1` to speed up early training

