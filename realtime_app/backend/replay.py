import threading
import time

import numpy as np

from realtime_app.config import PROCESSED_DIR, LABEL_MAP, BFI_FRAME_INTERVAL_MS, WINDOW_SIZE


def discover_replay_devices(processed_dir=None):
    """Return device folders that contain pvmatrix_*.npy files."""
    processed_dir = processed_dir or PROCESSED_DIR
    if not processed_dir.exists():
        return []
    return sorted([
        p.name for p in processed_dir.iterdir()
        if p.is_dir() and list(p.glob('pvmatrix_*.npy'))
    ])


def load_replay_clips(devices=None, processed_dir=None):
    """Load all pvmatrix_*.npy files grouped by label.

    Returns {'empty': [(arr, device, filename, timestamps), ...], 'occupied': [...]}
    Each arr is (W, T, K, C). timestamps is a 1D array of epoch times or None.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    devices = devices or discover_replay_devices(processed_dir)
    clips = {'empty': [], 'occupied': []}

    for device in devices:
        device_dir = processed_dir / device
        if not device_dir.is_dir():
            continue
        for fpath in sorted(device_dir.glob('pvmatrix_*.npy')):
            stem = fpath.stem.lower()
            label = None
            for key, val in LABEL_MAP.items():
                if key in stem:
                    label = 'empty' if val == 0 else 'occupied'
                    break
            if label is None:
                continue

            arr = np.load(fpath)

            # Try to load corresponding timestamps file
            # pvmatrix_empty_M7.npy -> ptimestamps_empty_M7.npy
            ts_name = fpath.name.replace('pvmatrix_', 'ptimestamps_')
            ts_path = device_dir / ts_name
            timestamps = None
            if ts_path.exists():
                timestamps = np.load(ts_path)

            clips[label].append((arr, device, fpath.name, timestamps))

    return clips


class ReplaySource:
    """Timestamp-based replay of recorded windows.

    Uses real pcap timestamps when available (from timestamps_*.npy files).
    Falls back to estimated timestamps using BFI_FRAME_INTERVAL_MS.

    Each window is played at its recorded timestamp, enabling accurate
    synchronization with video recordings taken during data capture.
    """

    def __init__(self, clips, on_window, speed=1.0):
        self.on_window = on_window
        self.speed = speed
        self._running = False
        self._thread = None
        self._paused = False

        # For tracking playback position
        self._elapsed_ms = 0
        self._total_elapsed_ms = 0  # never resets on loop
        self._wall_start = None

        # Build timeline: list of (timestamp_ms, arr, device, filename, label, window_idx)
        self._timeline = []
        self._build_timeline(clips)
        self._timeline_idx = 0

        # Track current filename for display
        self._current_filename = None

    @property
    def paused(self):
        return self._paused

    @property
    def elapsed_ms(self):
        return self._elapsed_ms

    @property
    def total_elapsed_ms(self):
        return self._total_elapsed_ms

    @property
    def current_filename(self):
        return self._current_filename

    @property
    def total_ms(self):
        if not self._timeline:
            return 0
        last_ts = self._timeline[-1][0]
        return last_ts + BFI_FRAME_INTERVAL_MS

    def _build_timeline(self, clips):
        """Build a flat timeline of all windows with timestamps."""
        # Sort clips by their first timestamp for proper ordering
        all_clips = []
        for label in ('empty', 'occupied'):
            for arr, device, filename, timestamps in clips.get(label, []):
                all_clips.append((arr, device, filename, label, timestamps))

        # Sort by first timestamp if available
        def get_first_ts(clip):
            ts = clip[4]  # timestamps array
            if ts is not None and len(ts) > 0:
                return ts[0]
            return 0
        all_clips.sort(key=get_first_ts)

        # Build timeline
        current_time_ms = 0.0
        for arr, device, filename, label, timestamps in all_clips:
            n_windows = arr.shape[0]

            if timestamps is not None and len(timestamps) >= n_windows:
                # Use real timestamps from pcap
                # Convert epoch seconds to milliseconds relative to first timestamp
                if len(self._timeline) == 0:
                    offset_ms = timestamps[0] * 1000.0
                else:
                    offset_ms = self._timeline[0][0] - self._timeline[0][0]  # align to start

                for window_idx in range(n_windows):
                    ts_ms = (timestamps[window_idx] * 1000.0) - (timestamps[0] * 1000.0)
                    self._timeline.append((
                        ts_ms,
                        arr,
                        device,
                        filename,
                        label,
                        window_idx,
                    ))
                current_time_ms = ts_ms + BFI_FRAME_INTERVAL_MS
            else:
                # Fall back to estimated timestamps
                for window_idx in range(n_windows):
                    self._timeline.append((
                        current_time_ms,
                        arr,
                        device,
                        filename,
                        label,
                        window_idx,
                    ))
                    current_time_ms += BFI_FRAME_INTERVAL_MS

    def set_speed(self, speed):
        """Change playback speed."""
        if speed <= 0:
            speed = 1
        self.speed = speed

    def set_paused(self, value=True):
        self._paused = value

    def toggle_paused(self):
        self.set_paused(not self._paused)

    def seek_to(self, timestamp_ms):
        """Seek to a specific timestamp in the timeline."""
        self._elapsed_ms = timestamp_ms
        for i, (ts, _, _, _, _, _) in enumerate(self._timeline):
            if ts >= timestamp_ms:
                self._timeline_idx = max(0, i - 1)
                break

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        last_emit_time = time.monotonic()
        last_tick = time.monotonic()

        while self._running:
            now = time.monotonic()
            wall_dt_ms = (now - last_tick) * 1000.0
            last_tick = now

            if self._paused:
                time.sleep(0.02)
                last_tick = time.monotonic()
                continue

            # Advance total elapsed time
            self._total_elapsed_ms += wall_dt_ms * self.speed

            # Find position within current loop
            total = self.total_ms if self.total_ms > 0 else 1
            self._elapsed_ms = self._total_elapsed_ms % total

            # Binary search for the right timeline index
            lo, hi = 0, len(self._timeline)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._timeline[mid][0] <= self._elapsed_ms:
                    lo = mid + 1
                else:
                    hi = mid
            self._timeline_idx = min(lo, len(self._timeline) - 1)
            self._current_filename = self._timeline[self._timeline_idx][4]

            # Emit prediction at most 20/sec
            if (now - last_emit_time) >= 0.05:
                idx = self._timeline_idx
                ts, arr, dev, fn, lbl, widx = self._timeline[idx]
                self.on_window(arr[widx], {
                    'true_label': lbl,
                    'replay_file': fn,
                    'replay_device': dev,
                    'video_time_s': self._total_elapsed_ms / 1000.0,
                    'elapsed_ms': ts,
                    'total_elapsed_ms': self._total_elapsed_ms,
                    'total_ms': self.total_ms,
                })
                last_emit_time = now

            # Sleep: skip at high speeds
            if self.speed <= 2:
                time.sleep(0.005)
