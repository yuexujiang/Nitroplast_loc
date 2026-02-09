"""
Training utilities for nitroplast model.

Includes:
- Training and validation loops
- Metrics computation
- Checkpointing
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_silhouette': [],
            'val_silhouette': [],
            'train_davies_bouldin': [],
            'val_davies_bouldin': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """Add metrics for current epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, path: str):
        """Save metrics history."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: str):
        """Load metrics history."""
        with open(path, 'r') as f:
            self.history = json.load(f)


def compute_embedding_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: [N, embedding_dim] - Protein embeddings
        labels: [N] - Binary labels (0 or 1)
    
    Returns:
        Dictionary with:
            - silhouette: Silhouette coefficient (-1 to 1, higher is better)
            - davies_bouldin: Davies-Bouldin index (lower is better)
    """
    metrics = {}
    
    try:
        # Silhouette score: measures how similar points are to their own cluster
        # vs other clusters. Range: [-1, 1], higher is better
        silhouette = silhouette_score(embeddings, labels)
        metrics['silhouette'] = silhouette
    except:
        metrics['silhouette'] = 0.0
    
    try:
        # Davies-Bouldin index: ratio of within-cluster to between-cluster distances
        # Lower is better (well-separated clusters)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        metrics['davies_bouldin'] = davies_bouldin
    except:
        metrics['davies_bouldin'] = float('inf')
    
    return metrics


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Handles variable-length sequences by batching them as lists.
    """
    sequences = [item['sequence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    ids = [item['id'] for item in batch]
    
    return {
        'sequences': sequences,
        'labels': labels,
        'ids': ids
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average loss for the epoch
        metrics: Dictionary of additional metrics
    """
    model.train()
    total_loss = 0.0
    all_embeddings = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        sequences = batch['sequences']
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(sequences)
            projected = outputs['projected']  # [batch_size, output_dim]
            
            # Compute contrastive loss
            loss = loss_fn(projected, labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Collect embeddings for metrics
        all_embeddings.append(projected.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute clustering metrics
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    metrics = compute_embedding_metrics(all_embeddings, all_labels)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model.
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary of additional metrics
    """
    model.eval()
    total_loss = 0.0
    all_embeddings = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        sequences = batch['sequences']
        labels = batch['labels'].to(device)
        
        outputs = model(sequences)
        projected = outputs['projected']
        
        loss = loss_fn(projected, labels)
        total_loss += loss.item()
        
        all_embeddings.append(projected.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    metrics = compute_embedding_metrics(all_embeddings, all_labels)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    load_path: str,
    device: str
) -> int:
    """
    Load model checkpoint.
    
    Returns:
        epoch: Epoch number to resume from
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    logger.info(f"Checkpoint loaded from {load_path} (epoch {epoch})")
    
    return epoch


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with differential learning rates.
    
    We use lower learning rate for pretrained ESM-2 layers.
    """
    # Separate parameters
    esm_params = []
    projector_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder.esm_model' in name:
            esm_params.append(param)
        else:
            projector_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': esm_params, 'lr': learning_rate * 0.1},  # 10x lower for ESM
        {'params': projector_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    scheduler_type: str = "cosine",
    min_lr: float = 1e-6
):
    """Create learning rate scheduler."""
    
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=min_lr
        )
    
    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.defaults['lr'],
            total_iters=num_training_steps
        )
    
    else:  # constant
        scheduler = None
    
    # Add warmup
    if num_warmup_steps > 0 and scheduler is not None:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[num_warmup_steps]
        )
    
    return scheduler


if __name__ == "__main__":
    # Test metrics computation
    np.random.seed(42)
    
    # Simulate good embeddings (well-separated clusters)
    good_embeddings = np.vstack([
        np.random.randn(50, 128) + np.array([2, 2]),  # Cluster 1
        np.random.randn(50, 128) + np.array([-2, -2])  # Cluster 2
    ])
    labels = np.array([1] * 50 + [0] * 50)
    
    metrics = compute_embedding_metrics(good_embeddings, labels)
    print("Good separation metrics:")
    print(f"  Silhouette: {metrics['silhouette']:.4f} (should be > 0.5)")
    print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f} (should be < 1.0)")
    
    # Simulate poor embeddings (overlapping clusters)
    poor_embeddings = np.random.randn(100, 128)
    
    metrics = compute_embedding_metrics(poor_embeddings, labels)
    print("\nPoor separation metrics:")
    print(f"  Silhouette: {metrics['silhouette']:.4f} (should be near 0)")
    print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f} (should be > 1.0)")
