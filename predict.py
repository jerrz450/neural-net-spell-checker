import torch
from models import TypoDetectorLSTM

class TypoDetector:

    def __init__(self, checkpoint_path, device=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.char_to_idx = checkpoint['char_to_idx']
        self.max_len = checkpoint['max_len']
        self.vocab_size = checkpoint['vocab_size']

        self.model = TypoDetectorLSTM(
            vocab_size=checkpoint['vocab_size'],
            n_embd=checkpoint.get('n_embd', 32),
            n_hidden=checkpoint.get('n_hidden', 128),
            n_layers=checkpoint.get('n_layers', 2)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, word):

        word = word.lower()
        word = ''.join(c for c in word if c.isalpha())

        indices = [self.char_to_idx.get(c, 0) for c in word[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))

        x = torch.tensor([indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()

        return prob

    def is_typo(self, word, threshold=0.5):
        prob = self.predict(word)
        return prob > threshold

    def predict_batch(self, words):
        results = []

        for word in words:
            prob = self.predict(word)
            results.append(prob)

        return results