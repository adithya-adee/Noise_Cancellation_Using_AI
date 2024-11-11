import os
import torch
from model import EnhancedNoiseReducer
from dataset import EnhancedNoiseDataset
from trainer import EnhancedNoiseReducerTrainer
import numpy as np
import json
from torch.utils.data import random_split

def create_directories(config):
    """Create necessary directories if they don't exist"""
    for directory in [config['model_save_dir'], config['results_dir']]:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Convert tensors or non-serializable objects to serializable types
def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, float):
        return float(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not serializable.")

def save_training_history(history, epoch, filepath):
    import json

    # Convert any np.float32 values to Python float
    converted_history = {k: float(v) if isinstance(v, np.float32) else v for k, v in history.items()}
    
    # Construct a dictionary to hold the history and epoch information
    data = {
        "epoch": epoch,
        "history": converted_history
    }
    
    # Save the training history and epoch to the specified file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)



def main():
    config = {
        'data_dir': '../data',
        'model_save_dir': '../models',
        'results_dir': '../results',
        'input_size': 2048,
        'hidden_size': 512,
        'learning_rate': 1e-6,
        'min_lr': 1e-6,
        'weight_decay': 0.01,
        'batch_size': 32,
        'num_epochs': 5,
        'train_split': 0.8,
        'n_fft': 2048,
        'n_mfcc': 13,
        'dropout_rate': 0.3,
        'clean_dir': '../data/new/clean/train',
        'noisy_dir': '../data/new/noisy/train',
        'noise_types_file': '../data/noise_types.json',
        'metrics_log_file': '../results/training_metrics.json',
        'grad_clip': 1.0,
        'patience': 5,
        'min_delta': 1e-4,
    }

    create_directories(config)

    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        print("Initializing dataset...")
        dataset = EnhancedNoiseDataset(
            config['clean_dir'],
            config['noisy_dir'],
            noise_types_file=config['noise_types_file'],
            n_mfcc=config['n_mfcc']
        )

        print("Splitting dataset...")
        train_size = int(config['train_split'] * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(
            dataset, 
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Training samples: {train_size}")
        print(f"Validation samples: {valid_size}")

        print("Initializing model...")
        model = EnhancedNoiseReducer(
            n_fft=config['n_fft'],
            n_mfcc=config['n_mfcc'],
            hidden_size=config['hidden_size']
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)

        print("Initializing trainer...")
        trainer = EnhancedNoiseReducerTrainer(model, train_dataset, valid_dataset, config)

        print("Starting training...")
        for epoch in range(config['num_epochs']):
            epoch_history = trainer.run_epoch(epoch)
            print(f"Epoch {epoch + 1}: {epoch_history}")

            # Save training history after each epoch
            save_training_history(
                epoch_history, epoch + 1, config['metrics_log_file']
            )

        print("Training completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
