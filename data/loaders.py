import json
import re
import random
import torch
from difflib import SequenceMatcher
from .typo_generator import add_typo

def load_brown_corpus(max_samples=50000):

    try:

        import nltk
        from nltk.corpus import brown

        try:
            words = brown.words()

        except:
            nltk.download('brown')
            words = brown.words()

        print(f"Loading Brown corpus...")

        typo_pairs = []
        seen = set()

        for word in words[:max_samples * 2]:
            word = word.lower()
            word = ''.join(c for c in word if c.isalpha())

            if len(word) < 2 or len(word) > 20:
                continue

            typo = add_typo(word)
            if typo != word:
                pair = (typo, word)

                if pair not in seen:
                    seen.add(pair)
                    typo_pairs.append(pair)

                    if len(typo_pairs) >= max_samples:
                        break 

        return typo_pairs

    except ImportError:
        print("ERROR: nltk not installed.")
        return []

def extract_word_typos(src_text, tgt_text):

    src_words = re.findall(r'\b[a-z]{2,20}\b', src_text.lower())
    tgt_words = re.findall(r'\b[a-z]{2,20}\b', tgt_text.lower())

    matcher = SequenceMatcher(None, src_words, tgt_words)
    typos = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == 'replace' and (i2-i1) == 1 and (j2-j1) == 1:
            src_word = src_words[i1]
            tgt_word = tgt_words[j1]

            if src_word != tgt_word and len(src_word) >= 2 and len(tgt_word) >= 2:
                typos.append((src_word, tgt_word))

    return typos

def load_github_typo_corpus(file_path, max_edits=50000):

    typo_pairs = []
    seen = set()

    print(f"Loading GitHub Typo Corpus from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:

        for line in f:
            try:
                data = json.loads(line)

                for edit in data.get('edits', []):
                    if not edit.get('is_typo', False):
                        continue

                    if edit.get('prob_typo', 0) < 0.8:
                        continue

                    src_text = edit.get('src', {}).get('text', '')
                    tgt_text = edit.get('tgt', {}).get('text', '')

                    word_typos = extract_word_typos(src_text, tgt_text)

                    for typo, correct in word_typos:

                        pair = (typo, correct)

                        if pair not in seen:

                            seen.add(pair)
                            typo_pairs.append(pair)

                            if len(typo_pairs) >= max_edits:
                                return typo_pairs

            except Exception:
                continue

    return typo_pairs

def filter_typos(typo_pairs):

    filtered = []

    for typo, correct in typo_pairs:
        if abs(len(typo) - len(correct)) > 3:
            continue

        similarity = SequenceMatcher(None, typo, correct).ratio()

        if 0.5 <= similarity <= 0.9:
            filtered.append((typo, correct))

    return filtered

def prepare_dataset(typo_pairs, max_len=20):

    labeled_words = []

    for typo, correct in typo_pairs:
        labeled_words.append((typo, 1))
        labeled_words.append((correct, 0))

    random.shuffle(labeled_words)

    all_text = ''.join([w for w, _ in labeled_words])
    chars = sorted(set(all_text))
    char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}
    char_to_idx['.'] = 0

    X, Y = [], []

    for word, label in labeled_words:
        
        indices = [char_to_idx.get(c, 0) for c in word[:max_len]]
        indices += [0] * (max_len - len(indices))
        
        X.append(indices)
        Y.append(label)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)

    n = len(X)

    n1 = int(0.8 * n)
    n2 = int(0.9 * n)

    return {
        'X_train': X[:n1],
        'Y_train': Y[:n1],
        'X_val': X[n1:n2],
        'Y_val': Y[n1:n2],
        'X_test': X[n2:],
        'Y_test': Y[n2:],
        'char_to_idx': char_to_idx,
        'vocab_size': len(char_to_idx),
        'max_len': max_len
    }
