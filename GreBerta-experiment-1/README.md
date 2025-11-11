# GreBerta Experiment

A demonstration project for the GreBerta model - a RoBERTa-based language model trained on Ancient Greek texts.

## About GreBerta

GreBerta is a state-of-the-art language model specifically designed for Classical Philology. It's a RoBERTa-base sized, monolingual, encoder-only model trained on Ancient Greek texts.

**Model**: [bowphs/GreBerta](https://huggingface.co/bowphs/GreBerta)

**Paper**: [Exploring Language Models for Classical Philology](https://arxiv.org/abs/2305.13698)

## Features

This demo showcases:
- **Masked Language Modeling**: Predict missing words in Ancient Greek sentences
- **Tokenization**: How the model processes Ancient Greek text
- **Model Information**: Architecture details and specifications
- **Multiple Prediction Methods**: Both pipeline-based and manual prediction approaches

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the demo script:

```bash
python greberta_demo.py
```

The script will:
1. Load the GreBerta model from Hugging Face
2. Demonstrate masked word prediction on Ancient Greek text
3. Show tokenization examples
4. Display model architecture information

## Examples

The demo includes predictions for phrases like:
- "ὁ [MASK] ἐστι καλός" (The [MASK] is beautiful)
- "ἐν ἀρχῇ ἦν ὁ [MASK]" (In the beginning was the [MASK])
- "σοφία ἐστι [MASK]" (Wisdom is [MASK])

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+

## Model Details

- **Type**: RoBERTa (Masked Language Model)
- **Architecture**: Encoder-only transformer
- **Size**: Base model (~125M parameters)
- **Language**: Ancient Greek (to 1453)
- **License**: Apache 2.0

## Citation

If you use GreBerta in your research, please cite:

```bibtex
@incollection{riemenschneiderfrank:2023,
    address = "Toronto, Canada",
    author = "Riemenschneider, Frederick and Frank, Anette",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL'23)",
    publisher = "Association for Computational Linguistics",
    title = "Exploring Large Language Models for Classical Philology",
    url = "https://arxiv.org/abs/2305.13698",
    year = "2023"
}
```

## References

- [GreBerta on Hugging Face](https://huggingface.co/bowphs/GreBerta)
- [Research Paper](https://arxiv.org/abs/2305.13698)
- [GitHub Repository](https://github.com/bowphs/BERT-ClassicalScholar)

