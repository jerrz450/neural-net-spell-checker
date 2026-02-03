import torch
import torch.nn as nn

class TypoDetectorLSTM(nn.Module):

    def __init__(self, vocab_size, n_embd=32, n_hidden=128, n_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_embd, padding_idx=0)

        self.lstm = nn.LSTM(
            n_embd,
            n_hidden,
            n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0,
            bidirectional=True
        )

        self.fc1 = nn.Linear(n_hidden * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        
        emb = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(emb)

        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        x = self.fc1(hidden_concat)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()
