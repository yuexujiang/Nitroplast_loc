"""
Data utilities for nitroplast protein dataset preparation.

Handles:
- FASTA loading and validation
- Sequence similarity-based splitting (to prevent data leakage)
- Dataset creation for PyTorch
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
import torch
from torch.utils.data import Dataset
import subprocess
import tempfile
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinSequenceDataset(Dataset):
    """PyTorch Dataset for protein sequences."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        ids: List[str],
        metadata: Optional[Dict] = None
    ):
        """
        Args:
            sequences: List of amino acid sequences
            labels: List of labels (1=nitroplast, 0=cytosolic)
            ids: List of protein IDs
            metadata: Optional dict with additional info (e.g., has_utp_motif)
        """
        assert len(sequences) == len(labels) == len(ids)
        self.sequences = sequences
        self.labels = labels
        self.ids = ids
        self.metadata = metadata or {}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'id': self.ids[idx]
        }
        
        # Add metadata if available
        if self.ids[idx] in self.metadata:
            item['metadata'] = self.metadata[self.ids[idx]]
        
        return item


def load_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """
    Load sequences from FASTA file.
    
    Returns:
        sequences: List of amino acid sequences
        ids: List of sequence identifiers
    """
    sequences = []
    ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return sequences, ids


def validate_sequences(
    sequences: List[str],
    min_length: int = 30,
    max_length: int = 2000
) -> Tuple[List[str], List[int]]:
    """
    Validate and filter protein sequences.
    
    Returns:
        valid_sequences: Filtered sequences
        valid_indices: Indices of valid sequences
    """
    valid_sequences = []
    valid_indices = []
    
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    for i, seq in enumerate(sequences):
        # Check length
        if not (min_length <= len(seq) <= max_length):
            continue
        
        # Check for valid amino acids (allow X for unknown)
        if not set(seq.upper()).issubset(standard_aa | {'X'}):
            continue
        
        valid_sequences.append(seq)
        valid_indices.append(i)
    
    logger.info(f"Validated {len(valid_sequences)}/{len(sequences)} sequences")
    return valid_sequences, valid_indices


def cluster_sequences_mmseqs(
    sequences: List[str],
    ids: List[str],
    similarity_threshold: float = 0.7,
    output_dir: Optional[str] = None
) -> Dict[int, List[int]]:
    """
    Cluster sequences by similarity using MMseqs2.
    
    This ensures that train/val/test splits are done at the cluster level,
    preventing information leakage through sequence homology.
    
    Args:
        sequences: List of protein sequences
        ids: List of sequence IDs
        similarity_threshold: Sequence identity threshold for clustering
        output_dir: Directory for temporary files
    
    Returns:
        clusters: Dict mapping cluster_id -> list of sequence indices
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Write sequences to FASTA
    fasta_path = output_dir / "sequences.fasta"
    with open(fasta_path, 'w') as f:
        for seq_id, seq in zip(ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")
    
    # Run MMseqs2 clustering
    db_path = output_dir / "seqDB"
    cluster_db = output_dir / "clusterDB"
    tmp_dir = output_dir / "tmp"
    
    try:
        # Create sequence database
        subprocess.run([
            "mmseqs", "createdb", str(fasta_path), str(db_path)
        ], check=True, capture_output=True)
        
        # Cluster sequences
        subprocess.run([
            "mmseqs", "cluster",
            str(db_path), str(cluster_db), str(tmp_dir),
            "--min-seq-id", str(similarity_threshold),
            "-c", "0.8",  # Coverage threshold
            "--cov-mode", "0"  # Bidirectional coverage
        ], check=True, capture_output=True)
        
        # Convert clustering to TSV
        tsv_path = output_dir / "clusters.tsv"
        subprocess.run([
            "mmseqs", "createtsv",
            str(db_path), str(db_path), str(cluster_db), str(tsv_path)
        ], check=True, capture_output=True)
        
        # Parse clustering results
        clusters = defaultdict(list)
        id_to_idx = {seq_id: idx for idx, seq_id in enumerate(ids)}
        
        with open(tsv_path, 'r') as f:
            for line in f:
                rep_id, member_id = line.strip().split('\t')
                cluster_id = id_to_idx[rep_id]  # Use representative as cluster ID
                member_idx = id_to_idx[member_id]
                clusters[cluster_id].append(member_idx)
        
        logger.info(f"Created {len(clusters)} sequence clusters")
        return dict(clusters)
    
    except subprocess.CalledProcessError as e:
        logger.error(f"MMseqs2 failed: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        logger.warning("MMseqs2 not found. Falling back to identity-based splitting.")
        return _fallback_clustering(sequences, ids)


def _fallback_clustering(
    sequences: List[str],
    ids: List[str]
) -> Dict[int, List[int]]:
    """
    Fallback clustering if MMseqs2 is not available.
    Simply assigns each sequence to its own cluster (no homology grouping).
    """
    logger.warning("Using identity-based splitting (no homology clustering)")
    return {i: [i] for i in range(len(sequences))}


def split_by_clusters(
    clusters: Dict[int, List[int]],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Split clusters into train/val/test sets while maintaining label distribution.
    
    This is crucial: we split at the CLUSTER level, not sequence level,
    to ensure test sequences are evolutionarily distinct from training.
    
    Returns:
        splits: Dict with keys 'train', 'val', 'test', each containing list of indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    
    # Group clusters by their label composition
    cluster_ids = list(clusters.keys())
    cluster_labels = []
    
    for cluster_id in cluster_ids:
        member_indices = clusters[cluster_id]
        member_labels = [labels[idx] for idx in member_indices]
        # Assign cluster label based on majority
        cluster_label = 1 if sum(member_labels) > len(member_labels) / 2 else 0
        cluster_labels.append(cluster_label)
    
    cluster_labels = np.array(cluster_labels)
    
    # Stratified split of clusters
    from sklearn.model_selection import train_test_split
    
    train_clusters, temp_clusters = train_test_split(
        cluster_ids,
        test_size=(val_ratio + test_ratio),
        stratify=cluster_labels,
        random_state=seed
    )
    
    temp_labels = cluster_labels[[cluster_ids.index(c) for c in temp_clusters]]
    val_clusters, test_clusters = train_test_split(
        temp_clusters,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=seed
    )
    
    # Expand clusters to sequence indices
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for cluster_id in train_clusters:
        splits['train'].extend(clusters[cluster_id])
    for cluster_id in val_clusters:
        splits['val'].extend(clusters[cluster_id])
    for cluster_id in test_clusters:
        splits['test'].extend(clusters[cluster_id])
    
    # Log statistics
    for split_name, indices in splits.items():
        split_labels = [labels[i] for i in indices]
        n_pos = sum(split_labels)
        n_neg = len(split_labels) - n_pos
        logger.info(f"{split_name}: {len(indices)} sequences "
                   f"(Positive: {n_pos}, Negative: {n_neg})")
    
    return splits


def prepare_datasets(
    positive_fasta: str,
    negative_fasta: str,
    output_dir: str,
    config: Dict,
    force_recompute: bool = False
) -> Tuple[ProteinSequenceDataset, ProteinSequenceDataset, ProteinSequenceDataset]:
    """
    Main function to prepare train/val/test datasets.
    
    Pipeline:
    1. Load positive and negative sequences
    2. Validate sequences
    3. Cluster by similarity (MMseqs2)
    4. Split clusters into train/val/test
    5. Create PyTorch datasets
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    splits_path = output_dir / "splits.npz"
    
    # Check if splits already exist
    if splits_path.exists() and not force_recompute:
        logger.info(f"Loading existing splits from {splits_path}")
        data = np.load(splits_path, allow_pickle=True)
        
        train_dataset = ProteinSequenceDataset(
            sequences=data['train_sequences'].tolist(),
            labels=data['train_labels'].tolist(),
            ids=data['train_ids'].tolist(),
            metadata=data['metadata'].item()
        )
        val_dataset = ProteinSequenceDataset(
            sequences=data['val_sequences'].tolist(),
            labels=data['val_labels'].tolist(),
            ids=data['val_ids'].tolist(),
            metadata=data['metadata'].item()
        )
        test_dataset = ProteinSequenceDataset(
            sequences=data['test_sequences'].tolist(),
            labels=data['test_labels'].tolist(),
            ids=data['test_ids'].tolist(),
            metadata=data['metadata'].item()
        )
        
        return train_dataset, val_dataset, test_dataset
    
    # Load sequences
    logger.info("Loading sequences from FASTA files...")
    pos_sequences, pos_ids = load_fasta(positive_fasta)
    neg_sequences, neg_ids = load_fasta(negative_fasta)
    
    # Combine and create labels
    all_sequences = pos_sequences + neg_sequences
    all_ids = pos_ids + neg_ids
    all_labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)
    
    # Validate
    logger.info("Validating sequences...")
    all_sequences, valid_indices = validate_sequences(
        all_sequences,
        min_length=config['data']['min_sequence_length'],
        max_length=config['data']['max_sequence_length']
    )
    all_ids = [all_ids[i] for i in valid_indices]
    all_labels = [all_labels[i] for i in valid_indices]
    
    # Cluster sequences
    logger.info("Clustering sequences by similarity...")
    clusters = cluster_sequences_mmseqs(
        all_sequences,
        all_ids,
        similarity_threshold=config['data']['sequence_similarity_threshold'],
        output_dir=output_dir / "clustering"
    )
    
    # Split by clusters
    logger.info("Splitting data...")
    splits = split_by_clusters(
        clusters,
        all_labels,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['seed']
    )
    
    # Create datasets
    def create_dataset(indices):
        return ProteinSequenceDataset(
            sequences=[all_sequences[i] for i in indices],
            labels=[all_labels[i] for i in indices],
            ids=[all_ids[i] for i in indices],
            metadata={}
        )
    
    train_dataset = create_dataset(splits['train'])
    val_dataset = create_dataset(splits['val'])
    test_dataset = create_dataset(splits['test'])
    
    # Save splits for reproducibility
    logger.info(f"Saving splits to {splits_path}")
    np.savez(
        splits_path,
        train_sequences=np.array(train_dataset.sequences, dtype=object),
        train_labels=np.array(train_dataset.labels),
        train_ids=np.array(train_dataset.ids, dtype=object),
        val_sequences=np.array(val_dataset.sequences, dtype=object),
        val_labels=np.array(val_dataset.labels),
        val_ids=np.array(val_dataset.ids, dtype=object),
        test_sequences=np.array(test_dataset.sequences, dtype=object),
        test_labels=np.array(test_dataset.labels),
        test_ids=np.array(test_dataset.ids, dtype=object),
        metadata=train_dataset.metadata
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="Force recompute")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_ds, val_ds, test_ds = prepare_datasets(
        positive_fasta=config['paths']['positive_fasta'],
        negative_fasta=config['paths']['negative_fasta'],
        output_dir=config['paths']['processed_dir'],
        config=config,
        force_recompute=args.force
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"Test: {len(test_ds)}")
