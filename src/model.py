import torch
import torch.nn as nn
from utils import calculate_psnr, calculate_snr

class EnhancedNoiseReducer(nn.Module):
    def __init__(self, n_fft=2048, n_mfcc=13, hidden_size=512):
        super(EnhancedNoiseReducer, self).__init__()
        
        spec_size = n_fft // 2 + 1
        total_input_size = spec_size + n_mfcc
        
        # Enhanced encoder with residual connections and layer normalization
        self.input_norm = nn.LayerNorm(total_input_size)
        
        # Define encoder sizes explicitly to ensure compatibility
        self.encoder_sizes = [
            (total_input_size, hidden_size),          # Layer 0: input -> hidden
            (hidden_size, hidden_size * 2),           # Layer 1: hidden -> hidden*2
            (hidden_size * 2, hidden_size * 2),       # Layer 2: hidden*2 -> hidden*2
            (hidden_size * 2, hidden_size * 2)        # Layer 3: hidden*2 -> hidden*2
        ]
        
        # Create encoder layers with explicit sizes
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.LayerNorm(out_size),
                nn.GELU(),
                nn.Dropout(0.3)
            ) for in_size, out_size in self.encoder_sizes
        ])
        
        # Define decoder sizes explicitly to match encoder sizes
        self.decoder_sizes = [
            (hidden_size * 2, hidden_size * 2),       # Layer 0: hidden*2 -> hidden*2
            (hidden_size * 2, hidden_size),           # Layer 1: hidden*2 -> hidden
            (hidden_size, hidden_size)                # Layer 2: hidden -> hidden
        ]
        
        # Create decoder layers with explicit sizes
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.LayerNorm(out_size),
                nn.GELU(),
                nn.Dropout(0.3)
            ) for in_size, out_size in self.decoder_sizes
        ])
        
        self.final_layer = nn.Linear(hidden_size, spec_size)
        self.output_activation = nn.Sigmoid()
        
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        
    def forward(self, x_spec, x_mfcc):
        batch_size, time_steps, freq_bins = x_spec.shape
        x_spec_flat = x_spec.view(-1, freq_bins)
        x_mfcc_flat = x_mfcc.view(-1, self.n_mfcc)
        
        # Combine and normalize inputs
        x = torch.cat([x_spec_flat, x_mfcc_flat], dim=1)
        x = self.input_norm(x)
        
        # Encoder with residual connections
        residuals = []
        for i, layer in enumerate(self.encoder_layers):
            residual = x
            x = layer(x)
            if i > 0 and x.size() == residual.size():  # Only add residual if sizes match
                x = x + residual
            residuals.append(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # Only add skip connection if the sizes match
            if i < len(residuals) - 1:
                skip_connection = residuals[-(i+2)]
                if x.size() == skip_connection.size():
                    x = x + skip_connection
        
        x = self.final_layer(x)
        x = self.output_activation(x)
        
        return x.view(batch_size, time_steps, -1)