from abc import ABC, abstractmethod
from typing import Tuple
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft
from scipy.spatial import ConvexHull
import spaudiopy as spa
from tqdm import tqdm

from utils import cart2sph, sph2cart, unpack_coordinates


def convert_A2B_format_tetramic(rirs_Aformat: NDArray) -> NDArray:

    if rirs_Aformat is None:
        print("[ERROR] Input rirs_Aformat is None")
        return None

    if rirs_Aformat.shape[1] != 4:
        print(f"[ERROR] Unexpected input shape: {rirs_Aformat.shape}, expected 4 channels")
        return None

    print(f"[DEBUG] Converting A2B, input shape = {rirs_Aformat.shape}")


    """
    Convert A format tetramic RIRs to B-format using SN3D normalisation and ACN ordering.

    Parameters
    ----------
    rirs_Aformat : NDArray
        RIRs in A-format, of shape (num_time_samples, num_channels).

    Returns
    -------
    NDArray
        RIRs in B format of shape (num_time_samples, num_channels) [w, y, z, x] ordering.
    """
    # Assume 4 unit vectors for tetrahedral mic (each row is [x, y, z])
    dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    #### WRITE YOUR CODE HERE ####

    # Create SN3D-normalized real SH basis functions (ACN order)
    # Order: [Y_0^0, Y_1^-1, Y_1^0, Y_1^1] => [W, Y, Z, X]

    # Stack SH functions into shape (num_mic_dirs, num_channels)

    # Invert to get A → B transform

    # Multiply wth inverted matrix with A-format RIRs to get B-format RIRs of
    # shape (num_time_samples, num_channels). Use einsum

    # Return B-format RIRs of shape: (num_time_samples, num_channels) in ACN/SN3D


    # Compute theta and phi for each direction
    z = dirs[:, 2]
    theta = np.arccos(z)  # θ ∈ [0, π]
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])  # ϕ ∈ [-π, π]

    # Compute SN3D-normalized real SH basis functions (ACN order)
    Y00 = 1 / np.sqrt(4 * np.pi)
    Y1m1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    Y10  = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
    Y11  = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)

    # Stack into (num_mics, num_channels=4) → columns are [Y_00, Y_1-1, Y_10, Y_11]
    Y = np.stack([Y00 * np.ones_like(Y1m1), Y1m1, Y10, Y11], axis=1)

    # Compute pseudo-inverse of Y: shape (4, 4)
    Y_pinv = np.linalg.pinv(Y)

    # Convert A-format to B-format using matrix multiplication
    # rirs_Aformat: (T, 4), Y_pinv.T: (4, 4)
    # Output: (T, 4) in order [w, y, z, x]
    rirs_Bformat = np.einsum('ij,tj->ti', Y_pinv, rirs_Aformat)

    return rirs_Bformat