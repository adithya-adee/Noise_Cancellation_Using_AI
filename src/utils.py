import torch
import numpy as np

def calculate_snr(clean, denoised):
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - denoised) ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def calculate_psnr(clean, denoised, max_value=1.0):
    mse = np.mean((clean - denoised) ** 2)
    return 20 * np.log10(max_value / (np.sqrt(mse) + 1e-8))


import librosa
import numpy as np

def compute_mfcc(audio, sr=16000, n_mfcc=13):
    """
    Compute MFCC (Mel Frequency Cepstral Coefficients) for a given audio signal.

    Parameters:
    - audio: numpy array, the audio signal.
    - sr: int, sample rate (default 16000).
    - n_mfcc: int, number of MFCC features to compute (default 13).

    Returns:
    - mfcc: numpy array of shape (n_mfcc, time_steps), the MFCC features.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc
