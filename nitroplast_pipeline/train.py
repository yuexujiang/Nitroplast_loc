"""
Main training script for nitroplast import code discovery.

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.esm_encoder import ESMEncoder
from models.projector import ProjectionHead, SupConLoss, NitroplastContrastiveModel
from utils.data_utils import prepare_datasets, collate_fn
from utils.training_utils import (
    train_epoch, validate, save_checkpoint,
    create_optimizer, create_scheduler,
    EarlyStopping, MetricsTracker
)
from utils.visualization import plot_training_curves, visualize_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    results_dir = Path(config['paths']['output_dir'])
    (results_dir / 'embeddings').mkdir(exist_ok=True, parents=True)
    (results_dir / 'plots').mkdir(exist_ok=True, parents=True)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        positive_fasta=config['paths']['positive_fasta'],
        negative_fasta=config['paths']['negative_fasta'],
        output_dir=config['paths']['processed_dir'],
        config=config,
        force_recompute=args.force_recompute
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Build model
    logger.info("Building model...")
    
    encoder = ESMEncoder(
        model_name=config['model']['esm_model_name'],
        use_lora=config['model']['use_lora'],
        lora_config={
            'r': config['model']['lora_r'],
            'alpha': config['model']['lora_alpha'],
            'dropout': config['model']['lora_dropout'],
            'target_modules': config['model']['lora_target_modules']
        },
        freeze_layers=config['model']['freeze_esm_layers'],
        pooling_method=config['model']['pooling_method'],
        device=device
    )
    
    projector = ProjectionHead(
        input_dim=config['model']['projector']['input_dim'],
        hidden_dims=config['model']['projector']['hidden_dims'],
        output_dim=config['model']['projector']['output_dim'],
        dropout=config['model']['projector']['dropout'],
        use_batch_norm=config['model']['projector']['use_batch_norm']
    )
    
    model = NitroplastContrastiveModel(encoder, projector).to(device)
    
    # Loss function
    loss_fn = SupConLoss(
        temperature=config['training']['temperature']
    )
    
    # Optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = create_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=config['training']['warmup_steps'],
        scheduler_type=config['training']['scheduler'],
        min_lr=config['training']['min_lr']
    )
    
    # Mixed precision scaler
    scaler = None
    if config['training']['mixed_precision'] and device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        mode='min'
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            grad_clip=config['training']['gradient_clip_norm'],
            scaler=scaler
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Train Silhouette: {train_metrics['silhouette']:.4f} | "
                   f"Val Silhouette: {val_metrics['silhouette']:.4f}")
        logger.info(f"Train DB: {train_metrics['davies_bouldin']:.4f} | "
                   f"Val DB: {val_metrics['davies_bouldin']:.4f}")
        
        # Update metrics tracker
        metrics_tracker.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_silhouette': train_metrics['silhouette'],
            'val_silhouette': val_metrics['silhouette'],
            'train_davies_bouldin': train_metrics['davies_bouldin'],
            'val_davies_bouldin': val_metrics['davies_bouldin']
        })
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                save_path=str(checkpoint_dir / 'best_model.pt')
            )
            logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
        
        # Regular checkpoint
        if epoch % config['training']['save_every_n_epochs'] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                save_path=str(checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info("Early stopping triggered")
            break
    
    # Save final metrics
    metrics_tracker.save(str(results_dir / 'training_metrics.json'))
    
    # Plot training curves
    plot_training_curves(
        metrics_history=metrics_tracker.history,
        output_path=str(results_dir / 'plots' / 'training_curves.png')
    )
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = validate(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device
    )
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Silhouette: {test_metrics['silhouette']:.4f}")
    logger.info(f"Test Davies-Bouldin: {test_metrics['davies_bouldin']:.4f}")
    
    # Compute and save embeddings for visualization
    logger.info("\nComputing final embeddings...")
    
    all_embeddings = []
    all_labels = []
    all_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences']
            labels = batch['labels']
            ids = batch['ids']
            
            embeddings = model.get_embeddings(sequences).cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())
            all_ids.extend(ids)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # Save embeddings
    np.savez(
        results_dir / 'embeddings' / 'test_embeddings.npz',
        embeddings=embeddings,
        labels=labels,
        ids=all_ids
    )
    
    # Visualize embeddings
    visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        ids=all_ids,
        output_path=str(results_dir / 'plots' / 'test_embeddings_tsne.png'),
        method='tsne',
        title='Test Set Protein Embeddings (t-SNE)'
    )
    
    if args.use_umap:
        visualize_embeddings(
            embeddings=embeddings,
            labels=labels,
            ids=all_ids,
            output_path=str(results_dir / 'plots' / 'test_embeddings_umap.png'),
            method='umap',
            title='Test Set Protein Embeddings (UMAP)'
        )
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nitroplast localization model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation of data splits'
    )
    parser.add_argument(
        '--use-umap',
        action='store_true',
        help='Also generate UMAP visualizations (requires umap-learn)'
    )
    
    args = parser.parse_args()
    main(args)
