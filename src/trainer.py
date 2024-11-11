import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import os
import numpy as np
from utils import calculate_psnr,calculate_snr

class EnhancedSTFTLoss(nn.Module):
    def __init__(self):
        super(EnhancedSTFTLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, output, target):
        magnitude_loss = self.l1_loss(torch.abs(output), torch.abs(target))
        # Add small epsilon to avoid log(0)
        phase_loss = 1 - torch.cos(torch.angle(output + 1e-8) - torch.angle(target + 1e-8)).mean()
        return magnitude_loss + 0.5 * phase_loss

class EnhancedNoiseReducerTrainer:
    def __init__(self, model, train_dataset, valid_dataset, config):
        self.config = config
        
        # Multi-core CPU setup
        self.num_workers = 4
        self.device = torch.device('cpu')
        self.model = DataParallel(model, device_ids=None)
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        # Initialize loss functions
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        self.stft_criterion = EnhancedSTFTLoss()
        
        # Calculate steps per epoch properly
        train_size = len(train_dataset)
        batch_size = min(self.config['batch_size'], train_size)
        steps_per_epoch = max(1, train_size // batch_size)
        
        # Optimizer setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        
        # Initialize metrics tracking
        self.metrics = {'train': {'loss': [], 'snr': [], 'psnr': []},
                       'valid': {'loss': [], 'snr': [], 'psnr': []}}
        
        self.best_val_loss = float('inf')
        self.best_val_snr = float('-inf')
        self.best_val_psnr = float('-inf')
        self.patience = 5
        self.patience_counter = 0

    def compute_loss(self, outputs, targets):
        """
        Compute the combined loss between model outputs and targets
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        Returns:
            Combined loss value
        """
        # MSE loss for overall reconstruction
        mse_loss = self.mse_criterion(outputs, targets)
        
        # L1 loss for better detail preservation
        l1_loss = self.l1_criterion(outputs, targets)
        
        # STFT loss for spectral reconstruction
        stft_loss = self.stft_criterion(outputs, targets)
        
        # Combine losses with weights
        total_loss = (
            0.3 * mse_loss +    # Basic reconstruction
            0.2 * l1_loss +     # Detail preservation
            0.5 * stft_loss     # Spectral accuracy
        )
        
        return total_loss

    def run_epoch(self, train=True):
        dataset = self.train_dataset if train else self.valid_dataset
        batch_size = min(self.config['batch_size'], len(dataset))
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        epoch_metrics = {'loss': 0.0, 'snr': 0.0, 'psnr': 0.0}
        num_batches = 0
        
        for batch in data_loader:
            noisy_specs = batch['noisy_spec'].to(self.device)
            clean_specs = batch['clean_spec'].to(self.device)
            noisy_mfcc = batch['noisy_mfcc'].to(self.device)
            
            if train:
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(noisy_specs, noisy_mfcc)
                loss = self.compute_loss(outputs, clean_specs)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(noisy_specs, noisy_mfcc)
                    loss = self.compute_loss(outputs, clean_specs)
            
            # Calculate metrics
            with torch.no_grad():
                snr = calculate_snr(clean_specs.cpu().numpy(), outputs.detach().cpu().numpy())
                psnr = calculate_psnr(clean_specs.cpu().numpy(), outputs.detach().cpu().numpy())
            
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['snr'] += snr
            epoch_metrics['psnr'] += psnr
            num_batches += 1
        
        # Average metrics over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics

    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_metrics = self.run_epoch(train=True)
            valid_metrics = self.run_epoch(train=False)
            
            print(f"Epoch [{epoch + 1}/{self.config['num_epochs']}]")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, SNR: {train_metrics['snr']:.2f}, PSNR: {train_metrics['psnr']:.2f}")
            print(f"Valid - Loss: {valid_metrics['loss']:.4f}, SNR: {valid_metrics['snr']:.2f}, PSNR: {valid_metrics['psnr']:.2f}")
            
            # Track improvements
            is_best = False
            if valid_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = valid_metrics['loss']
                is_best = True
                self.patience_counter = 0
            elif valid_metrics['snr'] > self.best_val_snr:
                self.best_val_snr = valid_metrics['snr']
                is_best = True
                self.patience_counter = 0
            elif valid_metrics['psnr'] > self.best_val_psnr:
                self.best_val_psnr = valid_metrics['psnr']
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save model
            if hasattr(self, 'save_model'):
                self.save_model(epoch, valid_metrics, is_best)
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print("Early stopping triggered! No improvement in any metric for 5 epochs.")
                break
            
            # Update metrics history
            for split in ['train', 'valid']:
                metrics = train_metrics if split == 'train' else valid_metrics
                for metric, value in metrics.items():
                    self.metrics[split][metric].append(value)
        
        return self.metrics

    # Define save_model in EnhancedNoiseReducerTrainer
    def save_model(self, epoch, metrics, is_best=False):
        """Save the model checkpoint"""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save current epoch model
        model_path = os.path.join(
            self.config['model_save_dir'], 
            f'model_epoch_{epoch+1}.pth'
        )
        torch.save(save_dict, model_path)
        
        # Save best model separately
        if is_best:
            best_model_path = os.path.join(
                self.config['model_save_dir'], 
                'best_model.pth'
            )
            torch.save(save_dict, best_model_path)