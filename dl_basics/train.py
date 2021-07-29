import numpy as np
import torch


def fit_eval_epoch(model, dataloader, criterion, optimizer=None, scorer=None, device='cpu'):
    losses, scores = [], []
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if optimizer is not None:
            model.train()
            optimizer.zero_grad()
            y_hat = model(X_batch)
            batch_loss = criterion(y_hat, y_batch)
            batch_loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                y_hat = model(X_batch)
                batch_loss = criterion(y_hat, y_batch)

        losses.append(batch_loss.item())
        if scorer is not None:
            batch_score = scorer(y_hat.detach(), y_batch.detach())
            scores.append(batch_score)

    return np.mean(losses), np.mean(scores)
