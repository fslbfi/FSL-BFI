import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Checkpoint
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_PATH = CHECKPOINT_DIR / 'exp2' / 'nb4_exp2_X7_encoder.pt'

# Data
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# Model constants
N_CHANNELS = 5
WINDOW_SIZE = 5
TARGET_SUBCARRIERS = 234
EMBED_DIM = 64

# BFI frame subtypes: Action=0x000d, Action-No-Ack=0x000e
BFI_SUBTYPES = {0x000d, 0x000e}

# Per-channel z-score normalization stats from raw V-matrices
# Layout: Re(ant0), Re(ant1), Re(ant2), Im(ant0), Im(ant1)
RAW_CHANNEL_MEAN = np.array([0.08629869, -0.21923323, 0.53889018, 0.05701223, 0.05199984])
RAW_CHANNEL_STD = np.array([0.42492233, 0.3399302, 0.20925189, 0.41827055, 0.36508108])

# Known devices (MAC -> config)
KNOWN_DEVICES = {
    '98:ee:94:99:d4:1a': {'name': 'M7',   'standard': 'AC', 'mimo': 'SU', 'config': '3x1', 'bw': 80},
    '9c:9e:d5:73:8a:2f': {'name': 'X7',   'standard': 'AX', 'mimo': 'SU', 'config': '3x2', 'bw': 80},
    '40:58:46:ea:91:fd': {'name': 'X300', 'standard': 'AX', 'mimo': 'SU', 'config': '3x2', 'bw': 80},
}

LABEL_MAP = {'empty': 0, 'stationary': 1, 'moving': 1}

# BFI frame interval in ms (typical: 10-20ms for Wi-Fi)
# Used to compute per-window timestamps for replay and video sync
BFI_FRAME_INTERVAL_MS = 10

DEFAULTS = {
    'window_size': WINDOW_SIZE,
    'capture_interface': os.environ.get('CAPTURE_INTERFACE') or 'wlan0mon',
    'mode': 'replay',  # 'live' or 'replay'
    'replay_devices': ['M7', 'X7', 'X300'],
    'replay_speed': 1.0,  # playback speed multiplier
    'video_offset_ms': 0,  # offset to align video with data
}

settings = dict(DEFAULTS)
