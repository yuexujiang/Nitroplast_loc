"""
Inference utilities for zero-shot protein localization prediction.

Uses trained model to predict nitroplast localization for novel proteins.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NitroplastPredictor:
    """
    Zero-shot predictor for nitroplast-localized proteins.
    
    Uses the learned embedding space to identify proteins similar
    to known nitroplast proteins.
    """
    
    def __init__(
        self,
        model,
        reference_embeddings: np.ndarray,
        reference_labels: np.ndarray,
        reference_ids: List[str],
        distance_metric: str = "cosine",
        confidence_threshold: float = 0.8,
        device: str = "cuda"
    ):
        """
        Args:
            model: Trained NitroplastContrastiveModel
            reference_embeddings: [N, embedding_dim] - Known protein embeddings
            reference_labels: [N] - Known labels (0 or 1)
            reference_ids: [N] - Reference protein IDs
            distance_metric: 'cosine' or 'euclidean'
            confidence_threshold: Minimum similarity to positive cluster
            device: cuda or cpu
        """
        self.model = model
        self.model.eval()
        self.device = device
        
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels
        self.reference_ids = reference_ids
        
        self.distance_metric = distance_metric
        self.confidence_threshold = confidence_threshold
        
        # Compute positive and negative centroids
        self.positive_mask = reference_labels == 1
        self.negative_mask = reference_labels == 0
        
        self.positive_centroid = reference_embeddings[self.positive_mask].mean(axis=0)
        self.negative_centroid = reference_embeddings[self.negative_mask].mean(axis=0)
        
        logger.info(f"Predictor initialized with {self.positive_mask.sum()} positive "
                   f"and {self.negative_mask.sum()} negative references")
    
    def compute_distance(
        self,
        query_embedding: np.ndarray,
        reference_embedding: np.ndarray
    ) -> float:
        """Compute distance between two embeddings."""
        if self.distance_metric == "cosine":
            # Cosine similarity (converted to distance)
            similarity = np.dot(query_embedding, reference_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(reference_embedding)
            )
            return 1.0 - similarity
        else:  # euclidean
            return np.linalg.norm(query_embedding - reference_embedding)
    
    def predict_single(
        self,
        sequence: str,
        protein_id: str
    ) -> Dict:
        """
        Predict localization for a single protein.
        
        Returns:
            Dictionary with:
                - protein_id: Protein identifier
                - prediction: 0 (cytosolic) or 1 (nitroplast)
                - confidence: Prediction confidence score
                - distance_to_positive: Distance to positive centroid
                - distance_to_negative: Distance to negative centroid
                - nearest_neighbors: Top 5 nearest reference proteins
        """
        # Get embedding
        with torch.no_grad():
            embedding = self.model.get_embeddings([sequence])[0].cpu().numpy()
        
        # Compute distances to centroids
        dist_to_positive = self.compute_distance(embedding, self.positive_centroid)
        dist_to_negative = self.compute_distance(embedding, self.negative_centroid)
        
        # Predict based on nearest centroid
        if dist_to_positive < dist_to_negative:
            prediction = 1
            confidence = 1.0 - dist_to_positive
        else:
            prediction = 0
            confidence = 1.0 - dist_to_negative
        
        # Find nearest neighbors
        distances = cdist(
            [embedding],
            self.reference_embeddings,
            metric='cosine' if self.distance_metric == 'cosine' else 'euclidean'
        )[0]
        
        nearest_indices = np.argsort(distances)[:5]
        nearest_neighbors = [
            {
                'id': self.reference_ids[idx],
                'label': int(self.reference_labels[idx]),
                'distance': float(distances[idx])
            }
            for idx in nearest_indices
        ]
        
        return {
            'protein_id': protein_id,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'distance_to_positive': float(dist_to_positive),
            'distance_to_negative': float(dist_to_negative),
            'nearest_neighbors': nearest_neighbors
        }
    
    def predict_batch(
        self,
        sequences: List[str],
        protein_ids: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict localization for multiple proteins.
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_ids = protein_ids[i:i+batch_size]
            
            for seq, pid in zip(batch_seqs, batch_ids):
                pred = self.predict_single(seq, pid)
                predictions.append(pred)
        
        return predictions
    
    def predict_from_fasta(
        self,
        fasta_path: str,
        output_path: Optional[str] = None,
        min_confidence: float = 0.7,
        return_top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Predict localizations for all proteins in a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            output_path: Where to save predictions (CSV)
            min_confidence: Only return predictions above this confidence
            return_top_k: If set, only return top K predictions
        
        Returns:
            DataFrame with predictions
        """
        # Load sequences
        sequences = []
        protein_ids = []
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            protein_ids.append(record.id)
        
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
        
        # Predict
        predictions = self.predict_batch(sequences, protein_ids)
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        
        # Add sequence length
        df['sequence_length'] = [len(s) for s in sequences]
        
        # Filter by confidence
        df = df[df['confidence'] >= min_confidence]
        
        # Filter for nitroplast predictions
        df_positive = df[df['prediction'] == 1].copy()
        
        # Sort by confidence
        df_positive = df_positive.sort_values('confidence', ascending=False)
        
        # Keep top K
        if return_top_k is not None:
            df_positive = df_positive.head(return_top_k)
        
        logger.info(f"Found {len(df_positive)} novel nitroplast candidates "
                   f"(confidence >= {min_confidence})")
        
        # Save
        if output_path is not None:
            df_positive.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return df_positive
    
    def cluster_predictions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        eps: float = 0.3,
        min_samples: int = 3
    ) -> np.ndarray:
        """
        Cluster novel predictions using DBSCAN.
        
        This can help identify groups of proteins with similar signals.
        
        Returns:
            cluster_labels: Cluster assignment for each protein
        """
        # Only cluster positive predictions
        positive_embeddings = embeddings[labels == 1]
        
        if len(positive_embeddings) < min_samples:
            logger.warning("Not enough positive predictions for clustering")
            return np.zeros(len(positive_embeddings), dtype=int)
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine' if self.distance_metric == 'cosine' else 'euclidean'
        )
        
        cluster_labels = clustering.fit_predict(positive_embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Identified {n_clusters} clusters ({n_noise} noise points)")
        
        return cluster_labels


def load_model_for_inference(
    checkpoint_path: str,
    config: Dict,
    device: str = "cuda"
):
    """
    Load trained model for inference.
    
    Returns:
        model: Loaded model
    """
    from models.esm_encoder import ESMEncoder
    from models.projector import ProjectionHead, NitroplastContrastiveModel
    
    # Recreate model architecture
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
    
    model = NitroplastContrastiveModel(encoder, projector)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    
    return model


def compute_reference_embeddings(
    model,
    dataset,
    batch_size: int = 32,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute embeddings for reference dataset.
    
    Returns:
        embeddings: [N, embedding_dim]
        labels: [N]
        ids: [N]
    """
    from torch.utils.data import DataLoader
    from utils.training_utils import collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    all_embeddings = []
    all_labels = []
    all_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequences']
            labels = batch['labels']
            ids = batch['ids']
            
            embeddings = model.get_embeddings(sequences).cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())
            all_ids.extend(ids)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels, all_ids


if __name__ == "__main__":
    # Test predictor with synthetic data
    np.random.seed(42)
    
    # Simulate reference embeddings
    ref_embeddings = np.vstack([
        np.random.randn(50, 128) + np.array([2, 0]),  # Positive
        np.random.randn(100, 128) + np.array([-2, 0])  # Negative
    ])
    ref_labels = np.array([1] * 50 + [0] * 100)
    ref_ids = [f"REF_{i}" for i in range(150)]
    
    print("Reference embeddings shape:", ref_embeddings.shape)
    print("Positive centroid:", ref_embeddings[ref_labels == 1].mean(axis=0)[:5])
    print("Negative centroid:", ref_embeddings[ref_labels == 0].mean(axis=0)[:5])
