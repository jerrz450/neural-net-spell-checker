import torch
from predict import TypoDetector
from collections import Counter

class SpellCorrector:

    def __init__(self, model_path, dictionary_path=None):

        self.detector = TypoDetector(model_path)
        self.dictionary = self._load_dictionary(dictionary_path)

    def _load_dictionary(self, path):

        if path:
            with open(path, 'r', encoding='utf-8') as f:
                return set(word.strip().lower() for word in f)

        else:
            try:
                import nltk
                from nltk.corpus import brown

                try:
                    words = brown.words()

                except:
                    nltk.download('brown')
                    words = brown.words()
                return set(w.lower() for w in words if w.isalpha())

            except:
                return set()

    def edits1(self, word):

        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):

        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def candidates(self, word):

        word = word.lower()
        if word in self.dictionary:
            return [word]

        candidates = self.edits1(word) & self.dictionary
        if not candidates:
            candidates = self.edits2(word) & self.dictionary

        return list(candidates) if candidates else [word]

    def correct(self, word, top_n=5):

        if not self.detector.is_typo(word):
            return word, []

        candidates = self.candidates(word)

        scored = []

        for candidate in candidates:
            score = sum(1 for a, b in zip(word, candidate) if a == b) / max(len(word), len(candidate))
            scored.append((candidate, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return word, [c for c, s in scored[:top_n]]

    def correct_text(self, text):

        words = text.split()
        corrected = []

        for word in words:

            clean_word = ''.join(c for c in word if c.isalpha())

            if clean_word:
                _, suggestions = self.correct(clean_word)

                if suggestions:
                    corrected.append(suggestions[0])
                else:
                    corrected.append(word)

            else:
                corrected.append(word)

        return ' '.join(corrected)


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage: python correct.py <model_path> [word_or_text]")
        print("Example: python correct.py spellchecker_models/lstm_brown_typo_detector.pt hlelo")
        sys.exit(1)

    model_path = sys.argv[1]
    corrector = SpellCorrector(model_path)

    if len(sys.argv) > 2:

        text = ' '.join(sys.argv[2:])
        words = text.split()

        for word in words:
            clean = ''.join(c for c in word if c.isalpha())

            if clean:
                original, suggestions = corrector.correct(clean)
                status = "TYPO" if suggestions else "CORRECT"
                sugg_str = ', '.join(suggestions[:3]) if suggestions else "-"
                print(f"{original:<15} | {status:<10} | {sugg_str}")
    else:

        while True:
            try:
                text = input("> ").strip()
                if text:
                    corrected = corrector.correct_text(text)
                    print(f"  Corrected: {corrected}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
