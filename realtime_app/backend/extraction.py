import numpy as np
import math
import cmath
from textwrap import wrap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import hex2dec, flip_hex
from bfi_angles import bfi_angles
from vmatrices import vmatrices

from realtime_app.config import (
    N_CHANNELS, TARGET_SUBCARRIERS, RAW_CHANNEL_MEAN, RAW_CHANNEL_STD
)


def get_subcarrier_idxs(standard, bw):
    if standard == 'AC':
        if bw == 80:
            subcarrier_idxs = np.arange(-122, 123)
            pilot_n_null = np.array([-104, -76, -40, -12, -1, 0, 1, 10, 38, 74, 102])
        elif bw == 40:
            subcarrier_idxs = np.arange(-58, 59)
            pilot_n_null = np.array([-54, -26, -12, -1, 0, 1, 10, 24, 52])
        elif bw == 20:
            subcarrier_idxs = np.arange(-28, 29)
            pilot_n_null = np.array([-21, -8, 0, 6, 21])
        else:
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
            neg_subcarriers = np.setdiff1d(np.arange(-122, 0, 2), np.arange(-118, -2, 4))
            pos_subcarriers = np.setdiff1d(np.arange(2, 124, 2), np.arange(6, 122, 4))
            return np.concatenate((neg_subcarriers, pos_subcarriers))
        else:
            return None
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
        return None
    tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit
    return Nc_users, Nr, order_angles, order_bits, tot_bits_users


def preprocess_vmatrix(v_mat):
    """Convert raw complex V-matrix to preprocessed real features matching notebook1.

    Raw vmatrices() output: complex (K, Nr, Nc) e.g. (234, 3, 1) for 3x1.
    Notebook1 pipeline:
      1. Truncate to TARGET_SUBCARRIERS, take first spatial stream (Nc=1)
      2. Squeeze: (K, Nr, 1) -> (K, Nr)
      3. Real/imag split: (K, 2*Nr) e.g. (234, 6)
      4. Drop Im(ant2) channel: (K, N_CHANNELS) e.g. (234, 5)
      5. Z-score normalize using pre-computed global stats

    Returns: float64 (K, N_CHANNELS) or None on failure.
    """
    try:
        if v_mat is None:
            return None

        # Truncate subcarriers and take first spatial stream only
        K = min(v_mat.shape[0], TARGET_SUBCARRIERS)
        v_trunc = v_mat[:K, :, :1]  # (K, Nr, 1)

        # Squeeze last dim
        vsq = v_trunc.squeeze(axis=-1)  # (K, Nr) complex

        # Real/imag split
        features = np.concatenate([vsq.real, vsq.imag], axis=-1)  # (K, 2*Nr)

        # Take first N_CHANNELS (drop Im(ant2) if Nr >= 3)
        if features.shape[1] < N_CHANNELS:
            return None
        features = features[:, :N_CHANNELS]  # (K, N_CHANNELS)

        # Z-score normalize using pre-computed global stats
        features = (features - RAW_CHANNEL_MEAN) / RAW_CHANNEL_STD

        return features.astype(np.float64)
    except Exception:
        return None


def extract_vmatrix_from_raw(packet, standard, mimo, config, NSUBC_VALID):
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
            return None

        if mimo == 'SU':
            psi_bit = 4 if codebook_info == '1' else 2
        elif mimo == 'MU':
            psi_bit = 7 if codebook_info == '1' else 5
        else:
            return None
        phi_bit = psi_bit + 2

        result = get_config_params(config, phi_bit, psi_bit)
        if result is None:
            return None
        Nc_users, Nr, order_angles, order_bits, tot_bits_users = result

        if standard == 'AX':
            Feedback_angles = packet[(i+62+2*int(config[-1])):(len(packet)-8)]
        elif standard == 'AC':
            Feedback_angles = packet[(i+58+2*int(config[-1])):(len(packet)-8)]

        Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
        Feedback_angles_bin = ''
        for j in range(len(Feedback_angles_splitted)):
            bin_str = str(format(hex2dec(Feedback_angles_splitted[j]), '08b'))
            bin_str = bin_str[::-1]
            Feedback_angles_bin += bin_str

        required_bits = tot_bits_users * NSUBC_VALID
        if len(Feedback_angles_bin) < required_bits:
            return None

        Feed_back_angles_bin_chunk = np.array(
            wrap(Feedback_angles_bin[:required_bits], tot_bits_users))
        angle = bfi_angles(Feed_back_angles_bin_chunk, True, NSUBC_VALID, order_bits)
        v_mat = vmatrices(angle, phi_bit, psi_bit, NSUBC_VALID, Nr, Nc_users, config)
        return preprocess_vmatrix(v_mat)
    except Exception:
        return None
