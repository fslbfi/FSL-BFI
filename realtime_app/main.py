import asyncio
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from realtime_app.config import settings, PROCESSED_DIR, CHECKPOINT_PATH, WINDOW_SIZE
from realtime_app.backend.model import load_encoder
from realtime_app.backend.prototypes import compute_precomputed_prototypes
from realtime_app.backend.inference import predict
from realtime_app.backend.windowing import WindowBuffer
from realtime_app.backend.capture import BFICapture
from realtime_app.backend.replay import load_replay_clips, discover_replay_devices
from realtime_app.backend.websocket import manager

LOG_MAX = 200

# Global state
encoder = None
proto_empty = None
proto_occupied = None
device = None
window_buffer = None
start_time = None
_loop = None

# Precomputed replay data
_replay_data = []  # list of dicts: {idx, timestamp_s, prediction, ground_truth, file, device, probabilities}
_replay_by_file = {}  # filename -> list of indices into _replay_data

# Log buffer
_log_buffer: deque = deque(maxlen=LOG_MAX)


def _log(level, message, source='system'):
    entry = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'level': level,
        'source': source,
        'message': message,
    }
    _log_buffer.append(entry)
    _broadcast({'type': 'log', 'data': entry})
    tag = {'info': 'INFO', 'warn': 'WARN', 'error': 'ERROR'}.get(level, level)
    print(f'[{tag}] [{source}] {message}')


def _broadcast(message):
    if _loop is None:
        return
    _loop.call_soon_threadsafe(asyncio.ensure_future, manager.broadcast(message))


def _on_vmatrix(vmatrix, timestamp_s=None):
    """Called by BFICapture for each extracted V-matrix (live mode only)."""
    if window_buffer is None:
        return
    window_buffer.push(vmatrix)
    if window_buffer.is_ready():
        window = window_buffer.get_window()
        try:
            result = predict(window, encoder, proto_empty, proto_occupied, device)
        except Exception as exc:
            _log('error', f'Prediction failed: {exc}', 'inference')
            return
        result['window_id'] = len(_replay_data)
        result['timestamp'] = time.time()
        result['video_time_s'] = timestamp_s
        result['true_label'] = None
        _broadcast({'type': 'prediction', 'data': result})


def _on_detect(ta, da, device_info):
    """Called by BFICapture when device is auto-detected."""
    _log('info', f'Detected device: {device_info["name"]} ({device_info["standard"]}/{device_info["config"]})', 'capture')
    _broadcast({'type': 'state', 'data': {
        'device_name': device_info['name'],
        'device_config': f'{device_info["standard"]}/{device_info["config"]}/{device_info["bw"]}MHz',
    }})


def _preload_replay_data(devices=None):
    """Preload all replay clips and run batch inference on all windows.

    Returns a list of dicts with predictions for every window.
    """
    global _replay_data, _replay_by_file

    clips = load_replay_clips(devices)
    n_empty = len(clips['empty'])
    n_occupied = len(clips['occupied'])
    if n_empty == 0 and n_occupied == 0:
        _log('error', 'No replay clips found', 'replay')
        return []

    _log('info', f'Preloading: {n_empty} empty, {n_occupied} occupied clips', 'replay')

    # Compute prototypes
    try:
        proto_e, proto_o = compute_precomputed_prototypes(PROCESSED_DIR, encoder, device)
    except Exception as exc:
        _log('error', f'Prototype computation failed: {exc}', 'replay')
        return []

    # Collect all windows with metadata
    all_windows = []
    metadata = []  # (filename, device, label, window_idx, total_windows_in_clip)

    for label in ('empty', 'occupied'):
        for arr, dev, fname, timestamps in clips.get(label, []):
            n_windows = arr.shape[0]
            for widx in range(n_windows):
                all_windows.append(arr[widx])
                metadata.append((fname, dev, label, widx, n_windows, timestamps))

    _log('info', f'Running batch inference on {len(all_windows)} windows...', 'replay')

    # Batch inference
    batch_size = 256
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = np.stack(all_windows[i:i + batch_size])
            x = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
            emb = encoder(x).cpu().numpy()
            all_embeddings.append(emb)
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Compute predictions from embeddings
    _replay_data = []
    _replay_by_file = {}
    current_time_ms = 0.0

    for i, embedding in enumerate(all_embeddings):
        fname, dev, label, widx, n_total, timestamps = metadata[i]

        # Compute distances
        dist_empty = float(((embedding - proto_e) ** 2).sum())
        dist_occupied = float(((embedding - proto_o) ** 2).sum())

        # Softmax
        neg_dists = np.array([-dist_empty, -dist_occupied])
        exp_dists = np.exp(neg_dists - neg_dists.max())
        probs = exp_dists / exp_dists.sum()

        # Use real timestamp if available
        if timestamps is not None and widx < len(timestamps):
            timestamp_s = float(timestamps[widx])
            if len(_replay_data) == 0:
                time_offset = timestamp_s
            timestamp_s -= time_offset
            current_time_ms = timestamp_s * 1000
        else:
            timestamp_s = current_time_ms / 1000.0
            current_time_ms += 10  # BFI_FRAME_INTERVAL_MS

        entry = {
            'idx': len(_replay_data),
            'timestamp_s': round(timestamp_s, 3),
            'prediction': 'Empty' if probs[0] > probs[1] else 'Occupied',
            'confidence': round(float(max(probs)), 4),
            'p_empty': round(float(probs[0]), 4),
            'p_occupied': round(float(probs[1]), 4),
            'dist_empty': round(dist_empty, 2),
            'dist_occupied': round(dist_occupied, 2),
            'ground_truth': label,
            'file': fname,
            'device': dev,
            'window_idx': widx,
        }
        _replay_data.append(entry)

        if fname not in _replay_by_file:
            _replay_by_file[fname] = []
        _replay_by_file[fname].append(entry['idx'])

    total_s = _replay_data[-1]['timestamp_s'] if _replay_data else 0
    _log('info', f'Preloaded {len(_replay_data)} windows ({total_s:.1f}s of data)', 'replay')

    return _replay_data


def _get_replay_slice(start_s, end_s):
    """Get predictions within a time range."""
    if not _replay_data:
        return []
    # Binary search for start
    lo = 0
    hi = len(_replay_data)
    while lo < hi:
        mid = (lo + hi) // 2
        if _replay_data[mid]['timestamp_s'] < start_s:
            lo = mid + 1
        else:
            hi = mid
    start_idx = lo
    # Find end
    while lo < len(_replay_data) and _replay_data[lo]['timestamp_s'] <= end_s:
        lo += 1
    return _replay_data[start_idx:lo]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, device, proto_empty, proto_occupied, start_time, _loop, window_buffer

    _loop = asyncio.get_running_loop()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log('info', f'Device: {device}', 'system')

    try:
        encoder = load_encoder(CHECKPOINT_PATH, device)
        _log('info', 'Encoder loaded', 'model')
    except Exception as exc:
        _log('error', f'Failed to load encoder: {exc}', 'model')
        encoder = None

    start_time = time.time()

    if encoder is not None:
        _preload_replay_data()

    yield


app = FastAPI(title='BFI Occupancy Detection', lifespan=lifespan)

frontend_dir = Path(__file__).parent / 'frontend'
app.mount('/css', StaticFiles(directory=str(frontend_dir / 'css')), name='css')
app.mount('/js', StaticFiles(directory=str(frontend_dir / 'js')), name='js')


@app.get('/')
async def index():
    return FileResponse(str(frontend_dir / 'index.html'))


@app.get('/api/state')
async def get_state():
    return {
        'mode': settings.get('mode', 'replay'),
        'capture_interface': settings['capture_interface'],
        'window_size': settings['window_size'],
        'uptime_s': int(time.time() - start_time) if start_time else 0,
        'replay_total_windows': len(_replay_data),
        'replay_total_s': _replay_data[-1]['timestamp_s'] if _replay_data else 0,
        'replay_files': list(_replay_by_file.keys()),
        'live_capturing': settings.get('live_capturing', False),
    }


@app.get('/api/replay')
async def get_replay_data():
    """Return all precomputed predictions."""
    return {'data': _replay_data, 'total_s': _replay_data[-1]['timestamp_s'] if _replay_data else 0}


@app.get('/api/logs')
async def get_logs():
    return {'logs': list(_log_buffer)}


@app.websocket('/ws/predictions')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        state = await get_state()
        state['settings'] = settings
        await websocket.send_json({'type': 'state', 'data': state})

        # Send all precomputed data upfront
        if _replay_data:
            await websocket.send_json({
                'type': 'replay_data',
                'data': {
                    'predictions': _replay_data,
                    'total_s': _replay_data[-1]['timestamp_s'],
                    'files': _replay_by_file,
                }
            })

        for entry in list(_log_buffer)[-50:]:
            await websocket.send_json({'type': 'log', 'data': entry})

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            await _handle_message(msg)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)


async def _handle_message(msg):
    msg_type = msg.get('type')
    data = msg.get('data', {})

    if msg_type == 'mode':
        mode = data.get('mode', 'replay')
        settings['mode'] = mode
        _log('info', f'Switching to mode: {mode}', 'system')
        await manager.broadcast({'type': 'state', 'data': await get_state()})

    elif msg_type == 'live':
        action = data.get('action')
        if action == 'start':
            global window_buffer
            window_buffer = WindowBuffer(settings['window_size'])
            iface = data.get('interface', settings['capture_interface'])
            settings['capture_interface'] = iface
            # Update device settings from frontend
            if 'device_name' in data:
                settings['device_name'] = data['device_name']
            if 'standard' in data:
                settings['standard'] = data['standard']
            if 'config' in data:
                settings['config'] = data['config']
            if 'bw' in data:
                settings['bw'] = data['bw']
            if 'mac' in data:
                settings['mac'] = data['mac']
            if 'mimo' in data:
                settings['mimo'] = data['mimo']
            source = BFICapture(_on_vmatrix, on_detect=_on_detect)
            source.start(iface)
            settings['live_capturing'] = True
            _log('info', f'Live capture started on {iface}', 'capture')
            await manager.broadcast({'type': 'state', 'data': await get_state()})
        elif action == 'stop':
            settings['live_capturing'] = False
            _log('info', 'Live capture stopped', 'capture')
            await manager.broadcast({'type': 'state', 'data': await get_state()})

    elif msg_type == 'settings':
        settings.update(data)
        _log('info', 'Settings updated', 'system')
        await manager.broadcast({'type': 'state', 'data': await get_state()})


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='BFI Occupancy Detection')
    parser.add_argument('--mode', choices=['live', 'replay'], default='replay')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--interface', default=None, help='Capture interface for live mode')
    args = parser.parse_args()

    settings['mode'] = args.mode
    if args.interface:
        settings['capture_interface'] = args.interface

    uvicorn.run("realtime_app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
