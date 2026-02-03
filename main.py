import os
import torch
from config import *
from data import load_brown_corpus, load_github_typo_corpus, filter_typos, prepare_dataset
from models import TypoDetectorLSTM
from utils import train_model, predict_typo

def main():

    print(f"Using device: {DEVICE}")
    print(f"Dataset: {USE_DATASET}")

    if USE_DATASET == "brown":

        typo_pairs = load_brown_corpus(max_samples=MAX_SAMPLES)
        typo_pairs_clean = typo_pairs

    elif USE_DATASET == "github":

        typo_pairs = load_github_typo_corpus(CORPUS_PATH, max_edits=MAX_SAMPLES)
        print(f"Loaded {len(typo_pairs)} typo pairs")

        typo_pairs_clean = filter_typos(typo_pairs)
        print(f"After filtering: {len(typo_pairs_clean)} pairs")

    else:
        print(f"ERROR: Unknown dataset '{USE_DATASET}'. Use 'brown' or 'github'")
        exit(1)

    data = prepare_dataset(typo_pairs_clean, max_len=MAX_LEN)

    print(f"\nDataset prepared:")
    print(f"  Train: {data['X_train'].shape}")
    print(f"  Val: {data['X_val'].shape}")
    print(f"  Test: {data['X_test'].shape}")
    print(f"  Vocab size: {data['vocab_size']}")

    model = TypoDetectorLSTM(
        vocab_size=data['vocab_size'],
        n_embd=32,
        n_hidden=128,
        n_layers=2
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, data, DEVICE, max_steps=MAX_STEPS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)

    save_dir = "spellchecker_models"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'char_to_idx': data['char_to_idx'],
        'vocab_size': data['vocab_size'],
        'max_len': data['max_len'],
        'n_embd': 32,
        'n_hidden': 128,
        'n_layers': 2
    }

    model_name = f'lstm_{USE_DATASET}_typo_detector.pt'
    save_path = os.path.join(save_dir, model_name)

    torch.save(checkpoint, save_path)
    print(f"\n Model saved to: {save_path}")

    test_words = [
        ("hello", "correct"),
        ("hlelo", "typo"),
        ("the", "correct"),
        ("teh", "typo"),
        ("implementation", "correct"),
        ("implimentation", "typo"),
        ("receive", "correct"),
        ("recieve", "typo"),
    ]

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"{'Word':<20} | {'Expected':<10} | {'Probability':<12} | {'Prediction':<10}")
    print("-"*70)

    correct = 0
    
    for word, expected in test_words:
        prob = predict_typo(word, model, data['char_to_idx'], data['max_len'], DEVICE)
        prediction = "TYPO" if prob > 0.5 else "CORRECT"
        is_correct = (prediction.lower() == expected)
        correct += is_correct

        print(f"{word:<20} | {expected:<10} | {prob:>12.4f} | {prediction:<10}`")

    print("-"*70)
    print(f"Accuracy: {correct}/{len(test_words)} = {100*correct/len(test_words):.1f}%")
    print("="*70)

if __name__ == "__main__":

    from predict import TypoDetector

    mode = 'predict'

    if mode  == "train":
        main()
    
    if mode == 'predict':

        checkpoint_path = r"spellchecker_models\lstm_brown_typo_detector.pt"
        typo_model = TypoDetector(checkpoint_path)
        predicted  = typo_model.is_typo('her')
        print(predicted)
