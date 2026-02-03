import torch
import torch.nn.functional as F

def train_model(model, data, device, max_steps=20000, batch_size=128, lr=0.001):

    Xtr = data['X_train'].to(device)
    Ytr = data['Y_train'].to(device)
    Xval = data['X_val'].to(device)
    Yval = data['Y_val'].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    best_val_loss = float('inf')
    best_step = 0

    for step in range(max_steps):

        model.train()

        ix = torch.randint(0, len(Xtr), (batch_size,), device=device)
        logits = model(Xtr[ix])
        loss = F.binary_cross_entropy_with_logits(logits, Ytr[ix].float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0:
            model.eval()

            with torch.no_grad():

                val_ix = torch.randint(0, len(Xval), (min(512, len(Xval)),), device=device)
                val_logits = model(Xval[val_ix])
                val_loss = F.binary_cross_entropy_with_logits(val_logits, Yval[val_ix].float())
                val_acc = ((torch.sigmoid(val_logits) > 0.5).long() == Yval[val_ix]).float().mean()

                if val_loss < best_val_loss:

                    best_val_loss = val_loss
                    best_step = step

            print(f"{step:6d}/{max_steps}: train={loss.item():.4f}, val={val_loss.item():.4f}, acc={val_acc.item():.4f}")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f} at step {best_step}")

    return model

def predict_typo(word, model, char_to_idx, max_len, device):

    model.eval()
    word = word.lower()
    word = ''.join(c for c in word if c.isalpha())

    indices = [char_to_idx.get(c, 0) for c in word[:max_len]]
    indices += [0] * (max_len - len(indices))

    x = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return prob
