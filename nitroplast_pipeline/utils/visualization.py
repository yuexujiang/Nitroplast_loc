"""
Visualization utilities for interpretability.

Includes:
- Attention map visualization
- t-SNE/UMAP embedding plots
- Motif discovery from high-attention regions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from Bio import SeqIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def visualize_attention_map(
    sequence: str,
    attention: np.ndarray,
    protein_id: str,
    output_path: str,
    top_k_residues: int = 20,
    window_size: Optional[int] = None
):
    """
    Visualize attention weights for a single protein.
    
    Args:
        sequence: Amino acid sequence
        attention: [seq_len, seq_len] attention matrix
        protein_id: Protein identifier
        output_path: Where to save the figure
        top_k_residues: Number of highest-attention residues to highlight
        window_size: If provided, show only this region around highest attention
    """
    seq_len = len(sequence)
    
    # Average attention each residue receives (column-wise mean)
    attention_scores = attention.mean(axis=0)
    
    # Find top-k residues
    top_indices = np.argsort(attention_scores)[-top_k_residues:]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Full attention matrix
    ax1 = plt.subplot(3, 2, (1, 3))
    im = ax1.imshow(attention, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('Residue Position', fontsize=12)
    ax1.set_ylabel('Residue Position', fontsize=12)
    ax1.set_title(f'Attention Matrix: {protein_id}', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Attention Weight')
    
    # 2. Per-residue attention scores
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(attention_scores, linewidth=1, alpha=0.7)
    ax2.scatter(top_indices, attention_scores[top_indices], 
                c='red', s=50, zorder=5, label=f'Top {top_k_residues}')
    ax2.set_xlabel('Residue Position', fontsize=12)
    ax2.set_ylabel('Average Attention', fontsize=12)
    ax2.set_title('Per-Residue Attention Profile', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sequence logo-style visualization of top residues
    ax3 = plt.subplot(3, 2, 4)
    top_residues = [(idx, sequence[idx], attention_scores[idx]) 
                    for idx in top_indices]
    top_residues.sort(key=lambda x: x[2], reverse=True)
    
    residue_text = []
    for idx, aa, score in top_residues[:10]:  # Show top 10
        residue_text.append(f"{aa}{idx+1} ({score:.3f})")
    
    ax3.text(0.1, 0.5, '\n'.join(residue_text), 
             fontsize=10, fontfamily='monospace',
             verticalalignment='center')
    ax3.axis('off')
    ax3.set_title('Top 10 Attended Residues', fontsize=12, fontweight='bold')
    
    # 4. Windowed view around highest attention
    if window_size:
        ax4 = plt.subplot(3, 2, (5, 6))
        max_idx = np.argmax(attention_scores)
        start = max(0, max_idx - window_size // 2)
        end = min(seq_len, max_idx + window_size // 2)
        
        window_attention = attention[start:end, start:end]
        window_seq = sequence[start:end]
        
        im2 = ax4.imshow(window_attention, cmap='YlOrRd', aspect='auto')
        ax4.set_xlabel('Local Position', fontsize=12)
        ax4.set_ylabel('Local Position', fontsize=12)
        ax4.set_title(f'Windowed View (±{window_size//2} around position {max_idx})',
                     fontsize=12, fontweight='bold')
        
        # Add sequence as x-tick labels
        if len(window_seq) <= 50:
            ax4.set_xticks(range(len(window_seq)))
            ax4.set_xticklabels(list(window_seq), fontsize=6, fontfamily='monospace')
        
        plt.colorbar(im2, ax=ax4, label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Attention map saved to {output_path}")


def extract_high_attention_regions(
    sequence: str,
    attention: np.ndarray,
    window_size: int = 15,
    min_attention: float = 0.1
) -> List[Dict]:
    """
    Extract sequence regions with high attention.
    
    These regions may contain novel targeting signals.
    
    Returns:
        List of dictionaries with:
            - start: Start position
            - end: End position
            - sequence: Subsequence
            - avg_attention: Average attention in this region
    """
    attention_scores = attention.mean(axis=0)
    seq_len = len(sequence)
    
    regions = []
    
    for i in range(seq_len - window_size + 1):
        window_attention = attention_scores[i:i+window_size].mean()
        
        if window_attention >= min_attention:
            regions.append({
                'start': i,
                'end': i + window_size,
                'sequence': sequence[i:i+window_size],
                'avg_attention': window_attention,
                'position': f"{i+1}-{i+window_size}"
            })
    
    # Sort by attention score
    regions.sort(key=lambda x: x['avg_attention'], reverse=True)
    
    return regions


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ids: List[str],
    output_path: str,
    method: str = "tsne",
    title: str = "Protein Embedding Space",
    metadata: Optional[Dict] = None
):
    """
    Visualize protein embeddings using dimensionality reduction.
    
    Args:
        embeddings: [N, embedding_dim]
        labels: [N] - Binary labels (0=cytosolic, 1=nitroplast)
        ids: [N] - Protein identifiers
        output_path: Where to save the figure
        method: 'tsne' or 'umap'
        title: Plot title
        metadata: Optional dict mapping id -> additional info (e.g., has_utp_motif)
    """
    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords = reducer.fit_transform(embeddings)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available, falling back to t-SNE")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords = reducer.fit_transform(embeddings)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by label
    colors = ['#3498db' if l == 0 else '#e74c3c' for l in labels]
    labels_str = ['Cytosolic' if l == 0 else 'Nitroplast' for l in labels]
    
    # If metadata available, distinguish uTP+ vs uTP-
    if metadata is not None:
        for i, protein_id in enumerate(ids):
            if labels[i] == 1:  # Nitroplast proteins
                has_utp = metadata.get(protein_id, {}).get('has_utp_motif', False)
                if has_utp:
                    colors[i] = '#e74c3c'  # Red for uTP+
                    labels_str[i] = 'Nitroplast (uTP+)'
                else:
                    colors[i] = '#f39c12'  # Orange for uTP- (hidden signal)
                    labels_str[i] = 'Nitroplast (uTP-)'
    
    # Plot
    for label in set(labels_str):
        mask = [l == label for l in labels_str]
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  c=[colors[i] for i, m in enumerate(mask) if m],
                  label=label, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Embedding visualization saved to {output_path}")


def analyze_attention_batch(
    model,
    sequences: List[str],
    ids: List[str],
    output_dir: str,
    layer_idx: int = -1,
    aggregate_heads: str = "mean",
    top_k: int = 20,
    window_size: int = 15
):
    """
    Batch analyze attention for multiple proteins.
    
    Generates individual attention maps and a summary CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_regions = []
    
    for seq, protein_id in zip(sequences, ids):
        # Get attention
        attention = model.encoder.get_attention_weights(
            [seq],
            layer_idx=layer_idx,
            aggregate_heads=aggregate_heads
        )
        attention = attention[0].cpu().numpy()  # [seq_len, seq_len]
        
        # Visualize
        vis_path = output_dir / f"{protein_id}_attention.png"
        visualize_attention_map(
            sequence=seq,
            attention=attention,
            protein_id=protein_id,
            output_path=str(vis_path),
            top_k_residues=top_k,
            window_size=window_size
        )
        
        # Extract high-attention regions
        regions = extract_high_attention_regions(
            sequence=seq,
            attention=attention,
            window_size=window_size,
            min_attention=0.1
        )
        
        for region in regions[:5]:  # Keep top 5 per protein
            region['protein_id'] = protein_id
            all_regions.append(region)
    
    # Save summary
    if all_regions:
        df = pd.DataFrame(all_regions)
        summary_path = output_dir / "high_attention_regions.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"High-attention regions saved to {summary_path}")


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    output_path: str
):
    """
    Plot training and validation curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, metrics_history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, metrics_history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Contrastive Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Silhouette score
    ax = axes[0, 1]
    ax.plot(epochs, metrics_history['train_silhouette'], label='Train', linewidth=2)
    ax.plot(epochs, metrics_history['val_silhouette'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Cluster Separation (Silhouette)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Davies-Bouldin index
    ax = axes[1, 0]
    ax.plot(epochs, metrics_history['train_davies_bouldin'], label='Train', linewidth=2)
    ax.plot(epochs, metrics_history['val_davies_bouldin'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title('Cluster Compactness (Davies-Bouldin)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    final_metrics = {
        'Train Loss': metrics_history['train_loss'][-1],
        'Val Loss': metrics_history['val_loss'][-1],
        'Val Silhouette': metrics_history['val_silhouette'][-1],
        'Val DB Index': metrics_history['val_davies_bouldin'][-1]
    }
    
    text = "Final Metrics:\n\n"
    for name, value in final_metrics.items():
        text += f"{name}: {value:.4f}\n"
    
    ax.text(0.1, 0.5, text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {output_path}")


if __name__ == "__main__":
    # Test visualization functions
    np.random.seed(42)
    
    # Simulate embeddings
    embeddings = np.vstack([
        np.random.randn(100, 128) + np.array([2, 2]),  # Nitroplast
        np.random.randn(150, 128) + np.array([-2, -2])  # Cytosolic
    ])
    labels = np.array([1] * 100 + [0] * 150)
    ids = [f"NP_{i}" for i in range(100)] + [f"CYT_{i}" for i in range(150)]
    
    # Test embedding visualization
    visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        ids=ids,
        output_path="/tmp/test_embeddings.png",
        method="tsne",
        title="Test Protein Embedding Space"
    )
    print("Test embedding visualization created at /tmp/test_embeddings.png")
