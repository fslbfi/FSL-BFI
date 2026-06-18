"""
Copyright (C) 2023 Khandaker Foysal Haque
contact: haque.k@northeastern.edu
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import os
import time
import subprocess
import json as _json
import numpy as np
import math
import threading
from textwrap import wrap
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from vmatrices import vmatrices
from bfi_angles import bfi_angles
from utils import hex2dec, flip_hex

LSB = True
_print_lock = threading.Lock()

def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)

# ─────────────────────────────────────────────────────────────
# DEVICE CONFIG
# ─────────────────────────────────────────────────────────────
DEVICES = {
    '98:ee:94:99:d4:1a': {
        'name':     'M7',
        'standard': 'AC',
        'mimo':     'SU',
        'config':   '3x1',
        'bw':        80,
    },
    '9c:9e:d5:73:8a:2f': {
        'name':     'X7',
        'standard': 'AX',
        'mimo':     'SU',
        'config':   '3x2',
        'bw':        80,
    },
    '40:58:46:ea:91:fd': {
        'name':     'X300',
        'standard': 'AX',
        'mimo':     'SU',
        'config':   '3x2',
        'bw':        80,
    }
}

# ─────────────────────────────────────────────────────────────
# TRACES
# ─────────────────────────────────────────────────────────────
TRACES = [
    'empty.pcapng',
    'stationary.pcapng',
    'moving.pcapng',
    'empty-2.pcapng',
    'stationary-2.pcapng',
    'moving-2.pcapng',
    'empty-3.pcapng',
    'stationary-3.pcapng',
    'moving-3.pcapng',
]

# BFI frame subtypes: Action=0x000d, Action-No-Ack=0x000e
BFI_SUBTYPES = {0x000d, 0x000e}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f'{m}m {s:02d}s' if m else f'{s}s'

def _ts():
    return datetime.now().strftime('%H:%M:%S')

def parse_timestamp(pkt):
    try:
        return float(pkt.frame_info.time_epoch)
    except (ValueError, TypeError, AttributeError):
        ts_str = str(pkt.frame_info.time_epoch).replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts_str)
        return dt.replace(tzinfo=timezone.utc).timestamp()

# ─────────────────────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────────────────────

def get_subcarrier_idxs(standard, bw):
    if standard == 'AC':
        if bw == 80:
            subcarrier_idxs = np.arange(-122, 123)
            pilot_n_null = np.array(
                [-104, -76, -40, -12, -1, 0, 1, 10, 38, 74, 102])
        elif bw == 40:
            subcarrier_idxs = np.arange(-58, 59)
            pilot_n_null = np.array([-54, -26, -12, -1, 0, 1, 10, 24, 52])
        elif bw == 20:
            subcarrier_idxs = np.arange(-28, 29)
            pilot_n_null = np.array([-21, -8, 0, 6, 21])
        else:
            print('input a valid bandwidth for IEEE 802.11ac')
            return None
        return np.setdiff1d(subcarrier_idxs, pilot_n_null)
    elif standard == 'AX':
        if bw == 160:
            subcarrier_idxs = np.arange(-1012, 1013, 4)
            pilot_n_null = np.array([-512, -8, -4, 0, 4, 8, 512])
            return np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 80:
            subcarrier_idxs = np.arange(-500, 504, 4)
            pilot_n_null = np.array([0])
            return np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 40:
            subcarrier_idxs = np.arange(-244, 248, 4)
            pilot_n_null = np.array([0])
            return np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 20:
            neg_subcarriers = np.setdiff1d(
                np.arange(-122, 0, 2), np.arange(-118, -2, 4))
            pos_subcarriers = np.setdiff1d(
                np.arange(2, 124, 2), np.arange(6, 122, 4))
            return np.concatenate((neg_subcarriers, pos_subcarriers))
        else:
            print('input a valid bandwidth for IEEE 802.11ax')
            return None
    print(f'Unknown standard: {standard}')
    return None

def get_config_params(config, phi_bit, psi_bit):
    if config == '4x2':
        Nc_users = 2; Nr = 4; phi_numbers = 5; psi_numbers = 5
        order_angles = [
            'phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41',
            'phi_22', 'phi_32', 'psi_32', 'psi_42'
        ]
        order_bits = [
            phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit,
            phi_bit, phi_bit, psi_bit, psi_bit
        ]
    elif config == '4x1':
        Nc_users = 1; Nr = 4; phi_numbers = 3; psi_numbers = 3
        order_angles = ['phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41']
        order_bits = [phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit]
    elif config == '3x3':
        Nc_users = 3; Nr = 3; phi_numbers = 3; psi_numbers = 3
        order_angles = ['phi_11', 'phi_21', 'psi_21', 'psi_31', 'phi_22', 'psi_32']
        order_bits = [phi_bit, phi_bit, psi_bit, psi_bit, phi_bit, psi_bit]
    elif config == '3x2':
        Nc_users = 2; Nr = 3; phi_numbers = 3; psi_numbers = 3
        order_angles = ['phi_11', 'phi_21', 'psi_21', 'psi_31', 'phi_22', 'psi_32']
        order_bits = [phi_bit, phi_bit, psi_bit, psi_bit, phi_bit, psi_bit]
    elif config == '3x1':
        Nc_users = 1; Nr = 3; phi_numbers = 2; psi_numbers = 2
        order_angles = ['phi_11', 'phi_21', 'psi_21', 'psi_31']
        order_bits = [phi_bit, phi_bit, psi_bit, psi_bit]
    else:
        print('Antenna configuration not supported:', config)
        return None
    tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit
    return Nc_users, Nr, order_angles, order_bits, tot_bits_users

def extract_vmatrix_from_raw(packet, standard, mimo, config, NSUBC_VALID):
    """Parse raw hex packet, return (v_matrix, bfi_angle_row) or (None, None)."""
    try:
        Header_length_dec = hex2dec(flip_hex(packet[4:8]))
        i = Header_length_dec * 2

        if standard == 'AX':
            packet_mimo_control = packet[(i+52):(i+62)]
            packet_mimo_control_binary = ''.join(
                format(int(c, 16), '04b') for c in flip_hex(packet_mimo_control))
            codebook_info = packet_mimo_control_binary[30]
        elif standard == 'AC':
            packet_mimo_control = packet[(i+52):(i+58)]
            packet_mimo_control_binary = ''.join(
                format(int(c, 16), '04b') for c in flip_hex(packet_mimo_control))
            codebook_info = packet_mimo_control_binary[13]
        else:
            return None, None

        if mimo == 'SU':
            psi_bit = 4 if codebook_info == '1' else 2
        elif mimo == 'MU':
            psi_bit = 7 if codebook_info == '1' else 5
        else:
            return None, None
        phi_bit = psi_bit + 2

        result = get_config_params(config, phi_bit, psi_bit)
        if result is None:
            return None, None
        Nc_users, Nr, order_angles, order_bits, tot_bits_users = result

        if standard == 'AX':
            Feedback_angles = packet[(i+62+2*int(config[-1])):(len(packet)-8)]
        elif standard == 'AC':
            Feedback_angles = packet[(i+58+2*int(config[-1])):(len(packet)-8)]

        Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
        Feedback_angles_bin = ''
        for j in range(len(Feedback_angles_splitted)):
            bin_str = str(format(hex2dec(Feedback_angles_splitted[j]), '08b'))
            if LSB:
                bin_str = bin_str[::-1]
            Feedback_angles_bin += bin_str

        required_bits = tot_bits_users * NSUBC_VALID
        if len(Feedback_angles_bin) < required_bits:
            return None, None

        Feed_back_angles_bin_chunk = np.array(
            wrap(Feedback_angles_bin[:required_bits], tot_bits_users))
        angle = bfi_angles(
            Feed_back_angles_bin_chunk, LSB, NSUBC_VALID, order_bits)
        v_mat = vmatrices(
            angle, phi_bit, psi_bit, NSUBC_VALID, Nr, Nc_users, config)
        return v_mat, angle
    except Exception as e:
        _tprint(f'[extract_vmatrix_from_raw] {type(e).__name__}: {e}')
        return None, None

# ─────────────────────────────────────────────────────────────
# MAIN PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def process_device(tmp_path, scenario, dev_cfg, data_dir, trim_start_s, trim_end_s):
    """
    Per-device BFI extraction pipeline: Pass 1 (scan) + Pass 1b (fetch raw) + Pass 2 (extract).
    Reads from a per-device temp file (already filtered by MAC + BFI subtype).
    All output is buffered and printed atomically at the end to avoid interleaving
    from concurrent device threads. Timing is reported per pass.
    """
    device    = dev_cfg['name']
    standard  = dev_cfg['standard']
    mimo      = dev_cfg['mimo']
    config    = dev_cfg['config']
    bw        = dev_cfg['bw']
    lines     = []  # buffered output, flushed atomically at end
    log       = lines.append

    t_device = time.time()

    # ── PASS 1: Scan — frame numbers and timestamps ──────────────────────────
    # The input temp file already contains only BFI Action frames for this device
    # (filtered by wlan.addr == <MAC> and subtype in {0x000d, 0x000e} upfront).
    # This scan pass extracts frame numbers and timestamps for trimming.
    # No tshark filter needed — the temp file is already device-specific.
    cmd_a = [
        'tshark', '-r', tmp_path,
        '-T', 'fields',
        '-e', 'frame.number',
        '-e', 'frame.time_epoch',
        '-e', 'wlan.fc.type_subtype',
        '-E', 'separator=|',
    ]

    bfi_frames = []  # (frame_number_str, timestamp_float)
    total_seen = 0
    t0 = time.time()

    try:
        proc = subprocess.Popen(
            cmd_a,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            total_seen += 1
            parts = line.split('|')
            if len(parts) < 3:
                continue
            frame_num, ts_raw, subtype_str = parts[0], parts[1], parts[2]
            try:
                subtype_val = int(subtype_str, 16)
            except ValueError:
                continue
            if subtype_val not in BFI_SUBTYPES:
                continue
            try:
                ts = float(ts_raw)
            except ValueError:
                ts_trunc = ts_raw[:26] + '+00:00' if ts_raw.endswith('Z') else ts_raw[:26]
                ts = datetime.fromisoformat(ts_trunc).replace(
                    tzinfo=timezone.utc).timestamp()
            bfi_frames.append((frame_num, ts))
        proc.wait()
    except Exception as e:
        log(f'  [{device}] Pass 1 error: {e}')
        _tprint('\n'.join(lines))
        return

    pass1_elapsed = _fmt_elapsed(time.time() - t0)
    log(f'  [{device}] Pass 1 scanning:  {total_seen} BFI frames [{pass1_elapsed}]')
    log(f'  [{device}] BFI frames found: {len(bfi_frames)}')

    if not bfi_frames:
        log(f'  [{device}] No BFI frames found.')
        _tprint('\n'.join(lines))
        return

    t_first = bfi_frames[0][1]
    t_last  = bfi_frames[-1][1]
    t_start = t_first + trim_start_s
    t_end   = t_last  - trim_end_s

    if t_start >= t_end:
        log(f'  [{device}] WARNING: Trim window empty — using full range.')
        t_start, t_end = t_first, t_last

    trimmed = [(fn, ts) for fn, ts in bfi_frames if t_start <= ts <= t_end]
    log(f'  [{device}] After trim ({trim_start_s}s/{trim_end_s}s): {len(trimmed)} packets '
        f'[{t_start - t_first:.1f}s – {t_end - t_first:.1f}s of session]')

    if not trimmed:
        _tprint('\n'.join(lines))
        return

    # ── PASS 1b: Fetch raw bytes for trimmed frames ───────────────────────────
    CHUNK_SIZE  = 500
    frame_nums  = [fn for fn, _ in trimmed]
    raw_by_frame = {}
    total_chunks = math.ceil(len(frame_nums) / CHUNK_SIZE)
    t0 = time.time()

    for i in range(0, len(frame_nums), CHUNK_SIZE):
        chunk      = frame_nums[i:i + CHUNK_SIZE]
        frame_set  = ','.join(chunk)
        cmd_b = [
            'tshark', '-r', tmp_path,
            '-Y', f'frame.number in {{{frame_set}}}',
            '-T', 'json',
            '-x',
        ]
        try:
            result_b = subprocess.run(cmd_b, capture_output=True, text=True)
            entries  = _json.loads(result_b.stdout) if result_b.stdout.strip() else []
            for entry in entries:
                try:
                    layers = entry['_source']['layers']
                    fn  = layers['frame']['frame.number']
                    raw = layers['frame_raw'][0]
                    raw_by_frame[fn] = raw
                except Exception:
                    continue
        except Exception as e:
            log(f'  [{device}] Pass 1b chunk error: {e}')

    pass1b_elapsed = _fmt_elapsed(time.time() - t0)
    raw_pool = [raw_by_frame[fn] for fn, _ in trimmed if fn in raw_by_frame]
    log(f'  [{device}] Pass 1b fetching: {total_chunks}/{total_chunks} chunks [{pass1b_elapsed}]')
    log(f'  ┌─────────────────────────────────────────────┐')
    log(f'  │ Packets entering V-matrix extraction: {len(raw_pool):>4} │')
    log(f'  └─────────────────────────────────────────────┘')

    # ── PASS 2: Extract V-matrices ────────────────────────────────────────────
    device_dir      = os.path.join(data_dir, device)
    os.makedirs(device_dir, exist_ok=True)
    saved_vmatrices = os.path.join(device_dir, f'vmatrix_{scenario}_{device}')
    bfa_dir         = os.path.join(device_dir, 'bfa')
    os.makedirs(bfa_dir, exist_ok=True)
    saved_bfas      = os.path.join(bfa_dir, f'bfa_{scenario}_{device}')
    log(f'  Output -> {saved_vmatrices}.npy')

    subcarrier_idxs = get_subcarrier_idxs(standard, bw)
    if subcarrier_idxs is None:
        _tprint('\n'.join(lines))
        return
    NSUBC_VALID = len(subcarrier_idxs)

    bfi_angles_all_packets = []
    v_matrices_all         = []
    skipped_extract        = 0
    t0 = time.time()

    for raw_pkt in raw_pool:
        v_mat, angle = extract_vmatrix_from_raw(raw_pkt, standard, mimo, config, NSUBC_VALID)
        if v_mat is not None:
            v_matrices_all.append(v_mat)
            bfi_angles_all_packets.append(angle)
        else:
            skipped_extract += 1

    pass2_elapsed = _fmt_elapsed(time.time() - t0)
    log(f'  [{device}] Pass 2 extracting: {len(raw_pool)}/{len(raw_pool)} pkt [{pass2_elapsed}]')

    if skipped_extract:
        log(f'  [{device}] Corrupted/skipped during extraction: {skipped_extract}')

    np.save(saved_vmatrices, np.array(v_matrices_all))
    np.save(saved_bfas, np.array(bfi_angles_all_packets))

    total_elapsed = _fmt_elapsed(time.time() - t_device)
    log(f'  [{device}] Saved {len(v_matrices_all)} V-matrices -> {saved_vmatrices}.npy')
    log(f'  [{device}] Total device time: {total_elapsed}')

    # Flush all output atomically — no interleaving with other device threads
    _tprint('\n'.join(lines))


# ─────────────────────────────────────────────────────────────
# PRE-FILTER: BFI SUBTYPE + MAC SPLIT (one temp file per device)
# ─────────────────────────────────────────────────────────────

def prefilter_device_bfi(trace_path, mac, device_name):
    """
    Extract BFI Action frames for a single device from the full PCAP into a
    temp file. Combines MAC address and BFI subtype filtering in one pass:
        wlan.addr == <MAC> and wlan.fc.type_subtype in {0x000d, 0x000e}
    All per-device passes then read their own smaller temp file instead of
    the full PCAP, eliminating redundant I/O during parallel processing.
    The caller is responsible for deleting the temp file.
    Returns the temp file path, or None if no BFI frames were found for this device.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.pcapng', delete=False)
    tmp.close()
    display_filter = 'wlan.addr == %s and wlan.fc.type_subtype in {0x000d,0x000e}' % mac
    cmd = [
        'tshark', '-r', trace_path,
        '-Y', display_filter,
        '-w', tmp.name,
    ]
    t0 = time.time()
    print(f' Pre-filtering BFI for {device_name} -> {tmp.name}')
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'tshark pre-filter failed (exit {result.returncode}):\n'
            + result.stderr.decode(errors='replace')
        )
    elapsed = _fmt_elapsed(time.time() - t0)
    size_mb = os.path.getsize(tmp.name) / 1024 / 1024
    orig_mb = os.path.getsize(trace_path) / 1024 / 1024
    # Only print size stats if the file has content
    if size_mb > 0:
        print(f' Pre-filter done [{elapsed}]  {orig_mb:.1f} MB -> {size_mb:.1f} MB '
              f'({100*size_mb/orig_mb:.1f}% of original)')
    else:
        print(f' Pre-filter done [{elapsed}]  {orig_mb:.1f} MB -> {size_mb:.1f} MB '
              f'(no BFI frames for {device_name})')
    return tmp.name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch Wi-Fi BFI extractor — pre-filtered by BFI subtype + MAC, parallel devices, timestamp-trimmed'
    )
    parser.add_argument('--dir', default='./traces2',
                        help='Directory of trace .pcapng files (default: ./traces2)')
    parser.add_argument('--data_dir', default='./data',
                        help='Output directory for .npy files (default: ./data)')
    parser.add_argument('--trim-start', default=30, type=float,
                        help='Seconds to trim from the start (default: 30)')
    parser.add_argument('--trim-end', default=30, type=float,
                        help='Seconds to trim from the end (default: 30)')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    print('=' * 60)
    print(f' Batch BFI Extractor  [pre-filter by BFI subtype + MAC split, parallel device mode]')
    print(f' Sampling   : full post-trim frames per device (no time-bin, no cross-device sync)')
    print(f' Trim start : {args.trim_start}s')
    print(f' Trim end   : {args.trim_end}s')
    print(f' Trace dir  : {args.dir}')
    print(f' Output dir : {args.data_dir}')
    print(f' Workers    : {len(DEVICES)} (one per device)')
    print('=' * 60)

    t_global = time.time()

    for trace in TRACES:
        trace_path = os.path.join(args.dir, trace)
        if not os.path.exists(trace_path):
            print(f'\nSkipping {trace}: not found')
            continue

        scenario = trace.replace('.pcapng', '')
        print(f'\n{"─" * 60}')
        print(f' Trace    : {trace}')
        print(f' Scenario : {scenario}')
        print(f'{"─" * 60}')

        # ── Pre-filter: parallel, one temp file per device, filtered by BFI subtype + MAC
        device_tmp_paths = {}
        with ThreadPoolExecutor(max_workers=len(DEVICES)) as prefilter_pool:
            prefilter_futures = {
                prefilter_pool.submit(
                    prefilter_device_bfi, trace_path, mac, dev_cfg['name']
                ): mac
                for mac, dev_cfg in DEVICES.items()
            }
            for future in as_completed(prefilter_futures):
                mac = prefilter_futures[future]
                device_tmp_paths[mac] = future.result()

        try:
            print(f' [{_ts()}] Starting {len(DEVICES)} devices in parallel...')

            with ThreadPoolExecutor(max_workers=len(DEVICES)) as pool:
                futures = {
                    pool.submit(
                        process_device,
                        device_tmp_paths[mac], scenario, dev_cfg,
                        args.data_dir, args.trim_start, args.trim_end
                    ): dev_cfg['name']
                    for mac, dev_cfg in DEVICES.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                        print(f' [{_ts()}] {name} done.')
                    except Exception as e:
                        print(f' [{_ts()}] {name} FAILED: {e}')
        finally:
            # Clean up all per-device temp files
            for mac, dev_cfg in DEVICES.items():
                tmp_path = device_tmp_paths.get(mac)
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    print(f' Temp file removed for {dev_cfg["name"]}: {tmp_path}')

        print(f'\n{"=" * 60}')

    print(f'\n{"=" * 60}')
    print(f' All done. Total elapsed: {_fmt_elapsed(int(time.time() - t_global))}')
    print(f' Output in: {args.data_dir}')
    print('=' * 60)