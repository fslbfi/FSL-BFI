import numpy as np
import torch
from pathlib import Path

from realtime_app.config import PROCESSED_DIR, LABEL_MAP
from realtime_app.backend.model import windows_to_tensor


def compute_precomputed_prototypes(processed_dir, encoder, device, exclude=None):
    exclude = set(exclude or ())
    all_windows = []
    all_labels = []

    for fpath in sorted(processed_dir.rglob('p*.npy')):
        if fpath.name in exclude:
            continue
        arr = np.load(fpath)
        stem = fpath.stem.lower()
        label = None
        for key, val in LABEL_MAP.items():
            if key in stem:
                label = val
                break
        if label is None:
            continue
        all_windows.append(arr)
        all_labels.extend([label] * arr.shape[0])

    if not all_windows:
        raise FileNotFoundError(f'No p*.npy files found under {processed_dir}')

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.array(all_labels)

    batch_size = 128
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i:i + batch_size]
            x = windows_to_tensor(batch, device)
            emb = encoder(x).cpu().numpy()
            embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)

    proto_empty = embeddings[all_labels == 0].mean(axis=0)
    proto_occupied = embeddings[all_labels == 1].mean(axis=0)

    return proto_empty, proto_occupied


def compute_kshot_prototypes(encoder, support_windows, support_labels, device):
    with torch.no_grad():
        x = windows_to_tensor(support_windows, device)
        embeddings = encoder(x).cpu().numpy()

    proto_empty = embeddings[support_labels == 0].mean(axis=0)
    proto_occupied = embeddings[support_labels == 1].mean(axis=0)

    return proto_empty, proto_occupied
