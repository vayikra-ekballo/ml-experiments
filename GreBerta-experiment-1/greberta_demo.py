"""
GreBerta Demo - Ancient Greek Language Model
Demonstrates masked language modeling with GreBerta
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch


def main():
    print("=" * 70)
    print("GreBerta - Ancient Greek Language Model Demo")
    print("=" * 70)
    print()
    
    # Determine device - use CPU to avoid MPS issues on Apple Silicon
    device = "cpu"
    print(f"Using device: {device}")
    print()
    
    # Load model and tokenizer
    print("Loading GreBerta model and tokenizer...")
    model_name = 'bowphs/GreBerta'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model = model.to(device)
    print("✓ Model loaded successfully!")
    print()
    
    # Create a fill-mask pipeline for easier usage
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=device)
    
    # Example 1: Simple masked prediction
    print("-" * 70)
    print("Example 1: Predicting masked words in Ancient Greek")
    print("-" * 70)
    
    # Ancient Greek sentences with masks
    examples = [
        "ὁ <mask> ἐστι καλός",  # "The [MASK] is beautiful"
        "ἐν ἀρχῇ ἦν ὁ <mask>",  # "In the beginning was the [MASK]"
        "σοφία ἐστι <mask>",     # "Wisdom is [MASK]"
    ]
    
    for i, text in enumerate(examples, 1):
        print(f"\nSentence {i}: {text}")
        print("Top 5 predictions:")
        
        try:
            predictions = fill_mask(text, top_k=5)
            for j, pred in enumerate(predictions, 1):
                token = pred['token_str']
                score = pred['score']
                sequence = pred['sequence']
                print(f"  {j}. Token: '{token}' | Score: {score:.4f}")
                print(f"     Full: {sequence}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Example 2: Manual token prediction
    print()
    print("-" * 70)
    print("Example 2: Manual masked token prediction")
    print("-" * 70)
    
    text = "καὶ ὁ λόγος σὰρξ <mask>"
    print(f"\nInput text: {text}")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Tokenized input IDs: {inputs['input_ids']}")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Find the masked token position
    mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    if len(mask_token_index) > 0:
        mask_token_index = mask_token_index[0]
        predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
        predicted_token = tokenizer.decode([predicted_token_id])
        
        print(f"Predicted token: {predicted_token}")
        
        # Get top 5 predictions
        top_5 = torch.topk(predictions[0, mask_token_index], 5)
        print("\nTop 5 predictions:")
        for idx, (score, token_id) in enumerate(zip(top_5.values, top_5.indices), 1):
            token = tokenizer.decode([token_id])
            print(f"  {idx}. '{token}' (score: {score:.4f})")
    
    # Example 3: Model information
    print()
    print("-" * 70)
    print("Example 3: Model Information")
    print("-" * 70)
    print(f"\nModel name: {model_name}")
    print(f"Model type: {model.config.model_type}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Max position embeddings: {model.config.max_position_embeddings}")
    
    # Example 4: Tokenization example
    print()
    print("-" * 70)
    print("Example 4: Tokenization of Ancient Greek Text")
    print("-" * 70)
    
    sample_text = "ἄνδρα μοι ἔννεπε μοῦσα"  # First words of the Odyssey
    print(f"\nOriginal text: {sample_text}")
    
    tokens = tokenizer.tokenize(sample_text)
    print(f"Tokens: {tokens}")
    
    token_ids = tokenizer.encode(sample_text)
    print(f"Token IDs: {token_ids}")
    
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded}")
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

