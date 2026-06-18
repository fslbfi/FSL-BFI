import numpy as np
import torch

from realtime_app.backend.model import windows_to_tensor


def predict(window, encoder, proto_empty, proto_occupied, device):
    with torch.no_grad():
        if window.ndim == 3:
            window = window[np.newaxis]
        x = windows_to_tensor(window, device)
        embedding = encoder(x).cpu().numpy()[0]

    dist_empty = float(((embedding - proto_empty) ** 2).sum())
    dist_occupied = float(((embedding - proto_occupied) ** 2).sum())

    neg_dists = np.array([-dist_empty, -dist_occupied])
    exp_dists = np.exp(neg_dists - neg_dists.max())
    probs = exp_dists / exp_dists.sum()

    return {
        'prediction': 'Empty' if probs[0] > probs[1] else 'Occupied',
        'confidence': float(max(probs)),
        'p_empty': float(probs[0]),
        'p_occupied': float(probs[1]),
        'dist_empty': dist_empty,
        'dist_occupied': dist_occupied,
    }
