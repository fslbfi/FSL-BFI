import subprocess
import threading
import time

from realtime_app.config import BFI_SUBTYPES, KNOWN_DEVICES
from realtime_app.backend.extraction import get_subcarrier_idxs, extract_vmatrix_from_raw


class BFICapture:
    """Stream BFI frames from tshark and extract V-matrices.

    Pipeline matches main_extract_batch.py:
      1. tshark captures with subtype filter (0x000d, 0x000e)
      2. Extract raw hex from each frame
      3. Parse V-matrix from raw hex
      4. Preprocess (real/imag, drop Im(ant2), normalize)
    """

    def __init__(self, on_vmatrix, on_detect=None):
        self.on_vmatrix = on_vmatrix
        self.on_detect = on_detect
        self._process = None
        self._thread = None
        self._running = False
        self._device_info = None

    def start(self, interface='wlan0mon'):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(interface,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._process = None

    def _run(self, interface):
        cmd = [
            'tshark', '-i', interface,
            '-T', 'fields',
            '-e', 'frame.number',
            '-e', 'frame.time_epoch',
            '-e', 'wlan.fc.type_subtype',
            '-e', 'wlan.ta',
            '-e', 'wlan.da',
            '-e', 'frame_raw',
            '-E', 'separator=|',
        ]

        while self._running:
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
                )
                for line in self._process.stdout:
                    if not self._running:
                        break
                    self._process_line(line)
                self._process.wait()
            except FileNotFoundError:
                time.sleep(2)
                continue
            except Exception:
                time.sleep(2)
                continue

    def _process_line(self, line):
        line = line.strip()
        if not line:
            return

        parts = line.split('|')
        if len(parts) < 6:
            return

        frame_num, ts_raw, subtype_str, ta, da, raw_hex = parts[:6]

        # Filter by BFI subtype
        try:
            subtype_val = int(subtype_str, 16)
        except (ValueError, TypeError):
            return
        if subtype_val not in BFI_SUBTYPES:
            return

        # Auto-detect device from first BFI frame
        if self._device_info is None and (ta or da):
            self._device_info = self._resolve_device(da) or self._resolve_device(ta)
            if self._device_info and self.on_detect:
                self.on_detect(ta, da, self._device_info)

        # Need raw hex and device info to extract V-matrix
        if not raw_hex or len(raw_hex) < 20:
            return
        if self._device_info is None:
            return

        dev = self._device_info
        nsubc = len(get_subcarrier_idxs(dev['standard'], dev['bw']) or [])
        if nsubc == 0:
            return

        vmatrix = extract_vmatrix_from_raw(
            raw_hex, dev['standard'], dev['mimo'], dev['config'], nsubc
        )
        if vmatrix is not None and self.on_vmatrix:
            # Parse real timestamp from tshark
            try:
                timestamp_s = float(ts_raw)
            except (ValueError, TypeError):
                timestamp_s = time.time()
            self.on_vmatrix(vmatrix, timestamp_s=timestamp_s)

    def _resolve_device(self, mac):
        if not mac:
            return None
        return KNOWN_DEVICES.get(mac.lower())
