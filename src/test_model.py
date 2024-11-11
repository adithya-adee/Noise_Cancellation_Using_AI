import os
import librosa
import numpy as np
import torch
import soundfile as sf  # Use soundfile instead of librosa.output
from model import NoiseReducer

class SpectralGating:
    def __init__(self, noise_reduction_factor=1.5):
        self.noise_reduction_factor = noise_reduction_factor
    
    def apply(self, noisy_signal):
        # Compute the spectrogram
        stft = np.abs(librosa.stft(noisy_signal))
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Estimate noise threshold
        noise_threshold = np.mean(magnitude) * self.noise_reduction_factor
        
        # Create mask to gate out noise
        mask = magnitude > noise_threshold
        gated_magnitude = magnitude * mask
        
        # Reconstruct the signal
        gated_stft = gated_magnitude * np.exp(1j * phase)
        denoised_signal = librosa.istft(gated_stft)
        
        return denoised_signal

def load_model(model_path, device):
    model = NoiseReducer(n_fft=2048, hidden_size=512)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def process_audio(input_file, output_file, model, spectral_gating, device):
    # Load noisy audio with a fixed duration to avoid length issues
    duration = 5.0  # Set a fixed duration in seconds
    noisy_signal, sr = librosa.load(input_file, sr=None, duration=duration)
    
    # Define the parameters
    n_fft = 2048
    hop_length = 512
    win_length = n_fft
    
    # Compute the spectrogram with explicit window length
    stft = librosa.stft(noisy_signal, n_fft=n_fft, hop_length=hop_length, 
                       win_length=win_length, window='hann')
    spectrogram = np.abs(stft)
    
    # Handle variable length input by processing in chunks
    chunk_size = 512
    total_chunks = (spectrogram.shape[1] + chunk_size - 1) // chunk_size
    processed_chunks = []
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, spectrogram.shape[1])
        
        # Extract chunk and pad if necessary
        chunk = spectrogram[:, start_idx:end_idx]
        if chunk.shape[1] < chunk_size:
            padding = np.zeros((chunk.shape[0], chunk_size - chunk.shape[1]))
            chunk = np.hstack((chunk, padding))
        
        # Process chunk - reshape to match model expectations
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        # Reshape to match the model's expected input
        chunk_tensor = chunk_tensor.permute(0, 2, 1)  # Shape: [1, 512, 1025]
        
        with torch.no_grad():
            processed_chunk = model(chunk_tensor)
            # Reshape back to original format
            processed_chunk = processed_chunk.permute(0, 2, 1)
        
        # Only keep the valid part (remove padding)
        valid_length = min(chunk_size, end_idx - start_idx)
        processed_chunk = processed_chunk.squeeze().cpu().numpy()[:, :valid_length]
        processed_chunks.append(processed_chunk)
    
    # Concatenate all processed chunks
    processed_spectrogram = np.concatenate(processed_chunks, axis=1)
    
    # Convert back to time domain
    denoised_signal = librosa.istft(processed_spectrogram, 
                                  hop_length=hop_length,
                                  win_length=win_length,
                                  window='hann')
    
    # Apply spectral gating to the denoised signal
    final_output = spectral_gating.apply(denoised_signal)
    
    # Save the final output using soundfile instead of librosa.output
    sf.write(output_file, final_output, sr)
    print(f"Processed audio saved to: {output_file}")

if __name__ == "__main__":
    # Configuration
    model_path = '../models/noise_reducer_model.pth'
    input_audio_file = '../data/new/noisy/test/83-11691-0028.wav'
    output_audio_file = '../denoised_output.wav'
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = load_model(model_path, device)
    
    # Initialize spectral gating
    spectral_gating = SpectralGating(noise_reduction_factor=1.5)
    
    # Process the audio file
    process_audio(input_audio_file, output_audio_file, model, spectral_gating, device)