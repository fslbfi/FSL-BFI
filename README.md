# FSL-BFI

Implementation of the paper **Binary Occupancy Detection: Utilizing Wi-BFI and Few-Shot Learning via Consumer Devices**.

This repository contains a GPL-3.0-licensed extraction script derived from [Wi-BFI](https://github.com/kfoysalhaque/Wi-BFI) and original MIT-licensed notebooks for downstream analysis. It also includes a real-time occupancy detection web app using Prototypical Networks.

---

## Getting Started

```bash
git clone https://github.com/fslbfi/FSL-BFI.git
cd FSL-BFI
```

### Setup Environment (uv)

This branch uses [uv](https://docs.astral.sh/uv/) for Python environment management. Install uv first:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then sync the environment:

```bash
uv sync
```

This downloads Python 3.10 (pinned via `.python-version`) and installs every dependency in `pyproject.toml`, resolved against `uv.lock`. Run anything inside the env with `uv run`:

```bash
uv run jupyter lab                 # launch Jupyter
uv run python main.py ...          # extraction tools
```

> **GPU users:** the default install is **CPU-only torch**. For CUDA, after `uv sync` run:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```

You may also want the following system packages for pcap capture:

```bash
sudo apt-get install tshark wireshark aircrack-ng
```

---

## Real-Time Occupancy Detection App

A web-based real-time app that captures live BFI frames from a Wi-Fi sniffer, processes them through the frozen ProtoNet encoder, and displays occupancy predictions (empty vs. occupied) on a responsive dashboard accessible from any device on your network.

### Architecture

```
Laptop (Sniffer)                    Phone/Tablet (Client)
+------------------+                +------------------+
| tshark capture   |   WiFi/LAN     |                  |
| V-matrix extract |  <---------->  |  Browser         |
| ProtoNet encoder |   WebSocket    |  Dashboard       |
| FastAPI server   |                |  (view only)     |
+------------------+                +------------------+
```

The **laptop running the app acts as the Wi-Fi sniffer** (monitor mode required). The phone/tablet is just a browser client viewing the dashboard -- no packet capture on the client side.

### Prerequisites

1. **Wi-Fi adapter in monitor mode** on the machine running the app:
   ```bash
   sudo airmon-ng check kill
   sudo airmon-ng start wlan0
   sudo iw dev wlan0mon set channel 64 80MHz
   ```

2. **tshark installed**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tshark

   # macOS
   brew install wireshark

   # Windows: install Wireshark (includes tshark)
   ```

3. **Trained encoder checkpoint** at `checkpoints/exp2/nb4_exp2_X7_encoder.pt`

4. **Processed training data** at `data/processed/` (for pre-computed prototypes)

### Running the App

```bash
# Using uv (recommended)
uv run python -m realtime_app.main

# Start in replay mode (default)
uv run python -m realtime_app.main --mode replay

# Start in live capture mode
uv run python -m realtime_app.main --mode live

# Directly, outside uv if the environment is already active
python -m realtime_app.main
```

For development, add `--reload` so Python changes are picked up automatically:

```bash
uv run python -m realtime_app.main --reload
```

| Flag | Example | Description |
|------|---------|-------------|
| `--mode` | `--mode live` | `live` (tshark capture) or `replay` (precomputed predictions). Default: `replay` |
| `--interface` | `--interface wlan0mon` | Capture interface for live mode |
| `--host` / `--port` | `--host 0.0.0.0 --port 8000` | Uvicorn bind address and port |
| `--reload` | `--reload` | Enable uvicorn auto-reload during development |

The server starts at `http://localhost:8000`. Open this URL in any browser on your network (e.g., `http://192.168.1.x:8000` from your phone).

### Two Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Replay** (default) | Preloads all 5,729 windows from processed data, runs batch inference at startup (~2s), streams precomputed predictions via WebSocket. Supports timeline scrubbing, variable speed (1x-100x), and video sync. | Hardware-free demos, recorded-data walkthroughs, projector presentations |
| **Live** | Captures BFI frames from a Wi-Fi adapter in monitor mode, extracts V-matrices in real-time, runs through the ProtoNet encoder window-by-window. | Live occupancy detection with a physical sniffer |

### Configurable Settings

All settings have defaults and can be changed via the UI or CLI:

| Setting | Default | Description |
|---------|---------|-------------|
| `window_size` | 5 | Consecutive BFI frames per window |
| `capture_interface` | wlan0mon | Network interface for sniffing |
| Device presets | M7/X7/X300 | Auto-fill MAC, standard, config, bandwidth, MIMO |
| Playback speed | 1x-100x | Adjustable replay speed |
| Video offset | 0ms | Sync offset for external video |

### Supported Devices

| Device | Standard | MIMO | Config | MAC |
|--------|----------|------|--------|-----|
| POCO M7 Pro 5G | 802.11ac | SU | 3x1 | 98:ee:94:99:d4:1a |
| POCO X7 Pro | 802.11ax | SU | 3x2 | 9c:9e:d5:73:8a:2f |
| vivo X300 Pro | 802.11ax | SU | 3x2 | 40:58:46:ea:91:fd |

### Timestamps for Video Sync

To enable real-time timestamps in replay mode (instead of estimated 10ms intervals):

```bash
# Single command: extracts timestamps from pcap + preprocesses to ptimestamps
uv run python prepare_timestamps.py --traces_dir D:/Thesis/data
```

This takes ~2-3 minutes and produces `ptimestamps_*.npy` files in `data/processed/`. The replay mode automatically picks these up for accurate timeline positioning and video synchronization.

### Docker (Optional)

```bash
docker compose up --build
```

The Docker container uses `network_mode: host` for packet capture access. The Wi-Fi adapter must be in monitor mode before starting.

### Dashboard Features

- Live prediction display (Empty / Occupied) with color-coded correctness indicator (checkmark/cross)
- Confidence percentage and probability bars (P(empty), P(occupied))
- Scatter chart (Chart.js) showing prediction history with ground truth labels and source file on hover
- Raw metrics row inline below chart (Euclidean distances, confidence, Correct/Wrong accuracy)
- File selector dropdown to filter chart by individual capture file (or view all)
- Timeline scrubber with play/pause/reset controls and window index display (e.g., "Window 1,234 of 5,729")
- Variable playback speed (1x-100x)
- Optional video sync with configurable offset
- System log panel with auto-scroll
- Left/right column layout: controls + prediction card (left), chart + log (right)
- Responsive design (desktop, tablet, mobile)

---

## Wi-BFI Usage (Batch Processing)

### 1. Prepare trace files

```bash
# check wireless interfaces
iw dev

# stop processes that revert the card to managed mode
sudo airmon-ng check kill

# enable monitor mode (example: wlan0)
sudo airmon-ng start wlp2s0

# tune channel & bandwidth
sudo iw dev wlp2s0mon set channel 64 80MHz
```

#### Capture examples

```bash
# onetime capture (save to PCAPNG)
sudo tshark -i wlp2s0mon -w - > capture_filename.pcapng
```

#### Display filters for devices

```text
# Poco M7
wlan.sa == 98:EE:94:99:D4:1A && (wlan.fc.type_subtype == 0x0d || wlan.fc.type_subtype == 0x0e)

# Poco X7
wlan.sa == 9C:9E:D5:73:8A:2F && (wlan.fc.type_subtype == 0x0d || wlan.fc.type_subtype == 0x0e)

# Vivo X300 Pro
wlan.sa == 40:58:46:EA:91:FD && (wlan.fc.type_subtype == 0x0d || wlan.fc.type_subtype == 0x0e)
```

### 2. Extract BFAs and reconstruct Vmatrices

- **Batch processing:**

```bash
python main_extract_batch.py \
    --dir /path/to/pcaps \         # default: ./traces2
    --data_dir /path/to/outputs    # default: ./data
```

- **Standalone invocation:**

```bash
python main.py \
    <trace_file> <standard> <mimo> <antenna-config> \
    <bandwidth> <mac_address> <packet_count> \
    <vmatrix_output> <bfa_output>

# example
python main.py 11ax_SU_4x2_160.pcapng AX SU 4x2 160 \
    20:c1:9b:fe:4f:ed 200 V_ax_su_4x2_160 bfa_ax_su_4x2_160
```

> **Arguments**
>  `<standard>`: `AC` / `AX`
>  `<mimo>`: `SU` / `MU`
>  `<antenna-config>`: `4x2`, `4x1`, `3x3`, `3x2`, `3x1`
>  `<bandwidth>`: `160` (AX only), `80`, `40`, `20`
>  `<mac_address>`: beamformee MAC
>  `<packet_count>`: how many packets to process (available)
>  Output files: names for Vmatrices and BFAs

---

## Project Structure

```
FSL-BFI/
├── realtime_app/              # Real-time detection web app
│   ├── main.py                # FastAPI entry point + WebSocket + precomputed replay
│   ├── config.py              # Settings and defaults
│   ├── backend/
│   │   ├── capture.py         # Live BFI capture (tshark)
│   │   ├── extraction.py      # V-matrix extraction
│   │   ├── model.py           # CNNEncoder + checkpoint loading
│   │   ├── prototypes.py      # Pre-computed + K-shot prototypes
│   │   ├── inference.py       # Prediction function
│   │   ├── windowing.py       # Sliding window buffer
│   │   ├── replay.py          # Historical replay source (loads pvmatrix + ptimestamps)
│   │   └── websocket.py       # WebSocket handler
│   └── frontend/
│       ├── index.html         # Responsive dashboard (Alpine.js + Chart.js)
│       ├── present.html       # Alternate replay presentation page
│       ├── css/styles.css     # Custom styles
│       └── js/
│           ├── app.js         # Alpine.js app + WebSocket client + Chart.js scatter plot
│           └── vendor/        # Vendored: alpine.min.js, chart.umd.min.js, tailwind.js
├── checkpoints/               # Trained model weights
│   └── exp2/                  # Cross-device encoders (the app loads nb4_exp2_X7_encoder.pt)
├── data/processed/            # Pre-processed window tensors (pvmatrix + ptimestamps)
├── notebooks/                 # Analysis notebooks (1-4)
├── prepare_timestamps.py      # Standalone: pcap → ptimestamps (single command for friend)
├── vmatrices.py               # V-matrix construction (GPL-3.0)
├── bfi_angles.py              # BFI angle extraction (GPL-3.0)
├── utils.py                   # Hex utilities (GPL-3.0)
├── main_extract_batch.py      # Batch BFI extraction (GPL-3.0)
├── main.py                    # Standalone extraction (GPL-3.0)
└── AGENTS.md                  # Agent context (this file)
```

---

## Licensing

This repository includes code modified from [Wi-BFI](https://github.com/kfoysalhaque/Wi-BFI), which is licensed under the **GNU General Public License v3.0**. The modified extraction script in this repository is therefore distributed under GPL-3.0 as well.

- `main_extract_batch.py` is derived from Wi-BFI and is licensed under GPL-3.0.
- The Jupyter notebooks in this repository are original work and are licensed under the MIT License. See `LICENSE-NOTEBOOKS`.
- The notebooks only use output files generated by the GPL-licensed extraction script and do not include or adapt Wi-BFI source code.

Please keep the corresponding license notices when redistributing or reusing any part of this repository.

## AI Assistance

Generative AI tools were used to assist with parts of the code development, debugging, and documentation in this repository. All final code, analysis, and written content were reviewed, edited, and validated by the authors before release.

This disclosure is provided for research transparency and does not change the licensing terms of the code or notebooks.

## Need Help?

For questions or support contact **Foysal Haque** via
[haque.k@northeastern.edu](mailto:haque.k@northeastern.edu)
or visit his [personal site](https://kfoysalhaque.github.io/).
