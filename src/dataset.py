import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

class EnhancedNoiseDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, noise_types_file=None, sr=16000, 
                 n_fft=2048, hop_length=512, fixed_time_dim=400, n_mfcc=13):
        """
        Initialize the dataset
        Args:
            clean_dir (str): Directory containing clean audio files
            noisy_dir (str): Directory containing noisy audio files
            noise_types_file (str, optional): JSON file mapping filenames to noise types
            sr (int): Sample rate
            n_fft (int): FFT size
            hop_length (int): Hop length for STFT
            fixed_time_dim (int): Fixed time dimension for all spectrograms
            n_mfcc (int): Number of MFCC coefficients
        """
        super().__init__()
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_time_dim = fixed_time_dim
        self.n_mfcc = n_mfcc

        # Load noise types if provided
        self.noise_types = {}
        if noise_types_file and os.path.exists(noise_types_file):
            with open(noise_types_file, 'r') as f:
                self.noise_types = json.load(f)

        # Get file pairs
        self.file_pairs = self._load_file_pairs()
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"No matching files found in {clean_dir} and {noisy_dir}")

    def _load_file_pairs(self):
        """Load matching pairs of clean and noisy audio files"""
        clean_files = sorted([f for f in os.listdir(self.clean_dir) 
                            if f.endswith(('.wav', '.WAV', '.flac', '.FLAC'))])
        noisy_files = sorted([f for f in os.listdir(self.noisy_dir) 
                            if f.endswith(('.wav', '.WAV', '.flac', '.FLAC'))])
        
        # Create pairs of files that exist in both directories
        pairs = []
        for clean_file in clean_files:
            base_name = os.path.splitext(clean_file)[0]
            # Look for matching noisy file with any supported extension
            matching_noisy = [f for f in noisy_files 
                            if os.path.splitext(f)[0] == base_name]
            if matching_noisy:
                pairs.append((
                    os.path.join(self.clean_dir, clean_file),
                    os.path.join(self.noisy_dir, matching_noisy[0])
                ))
        
        return pairs

    def _normalize(self, audio):
        """Normalize audio to zero mean and unit variance"""
        audio = audio.astype(np.float32)
        return (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

    def _extract_mfcc(self, audio):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return self._pad_or_crop_2d(mfcc, self.fixed_time_dim)

    def _pad_or_crop_2d(self, feature, target_length):
        """Pad or crop 2D feature to target length"""
        curr_length = feature.shape[1]
        if curr_length < target_length:
            padding = target_length - curr_length
            return np.pad(feature, ((0, 0), (0, padding)), mode='constant')
        return feature[:, :target_length]

    def _get_noise_type(self, filename):
        """Get noise type from filename"""
        basename = os.path.basename(filename)
        return self.noise_types.get(basename, 'unknown')

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        try:
            clean_path, noisy_path = self.file_pairs[idx]
            
            # Load audio files
            clean_audio, _ = librosa.load(clean_path, sr=self.sr)
            noisy_audio, _ = librosa.load(noisy_path, sr=self.sr)

            # Ensure both audio files have the same length
            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]

            # Normalize audio
            clean_audio = self._normalize(clean_audio)
            noisy_audio = self._normalize(noisy_audio)

            # Compute spectrograms
            clean_spec = librosa.stft(clean_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            noisy_spec = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)

            # Convert to magnitude spectrograms
            clean_spec_mag = np.abs(clean_spec)
            noisy_spec_mag = np.abs(noisy_spec)

            # Extract MFCC features
            clean_mfcc = self._extract_mfcc(clean_audio)
            noisy_mfcc = self._extract_mfcc(noisy_audio)

            # Ensure fixed dimensions
            clean_spec_mag = self._pad_or_crop_2d(clean_spec_mag, self.fixed_time_dim)
            noisy_spec_mag = self._pad_or_crop_2d(noisy_spec_mag, self.fixed_time_dim)

            # Get noise type
            noise_type = self._get_noise_type(noisy_path)

            return {
                'noisy_spec': torch.FloatTensor(noisy_spec_mag.T),
                'clean_spec': torch.FloatTensor(clean_spec_mag.T),
                'noisy_mfcc': torch.FloatTensor(noisy_mfcc.T),
                'clean_mfcc': torch.FloatTensor(clean_mfcc.T),
                'noise_type': noise_type,
                'clean_path': clean_path,
                'noisy_path': noisy_path
            }
        except Exception as e:
            print(f"Error processing files at index {idx}: {e}")
            print(f"Clean path: {clean_path}")
            print(f"Noisy path: {noisy_path}")
            raise e

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.file_pairs)