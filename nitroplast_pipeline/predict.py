"""
Prediction script for zero-shot nitroplast localization discovery.

Usage:
    python predict.py --checkpoint results/checkpoints/best_model.pt \
                      --proteome data/raw/full_proteome.fasta \
                      --output results/predictions/
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from utils.inference_utils import (
    NitroplastPredictor,
    load_model_for_inference,
    compute_reference_embeddings
)
from utils.data_utils import load_fasta
from utils.visualization import visualize_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load trained model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device
    )
    
    # Compute reference embeddings (from training/validation data)
    logger.info("Computing reference embeddings...")
    
    if args.reference_embeddings is not None:
        # Load precomputed reference embeddings
        logger.info(f"Loading reference embeddings from {args.reference_embeddings}")
        data = np.load(args.reference_embeddings)
        ref_embeddings = data['embeddings']
        ref_labels = data['labels']
        ref_ids = data['ids'].tolist()
    else:
        # Compute from reference FASTA files
        logger.info("Computing reference embeddings from FASTA files...")
        
        # Load reference sequences
        pos_sequences, pos_ids = load_fasta(config['paths']['positive_fasta'])
        neg_sequences, neg_ids = load_fasta(config['paths']['negative_fasta'])
        
        all_sequences = pos_sequences + neg_sequences
        all_ids = pos_ids + neg_ids
        all_labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)
        
        # Compute embeddings in batches
        all_embeddings = []
        batch_size = args.batch_size
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(all_sequences), batch_size):
                batch_seqs = all_sequences[i:i+batch_size]
                embeddings = model.get_embeddings(batch_seqs).cpu().numpy()
                all_embeddings.append(embeddings)
        
        ref_embeddings = np.vstack(all_embeddings)
        ref_labels = np.array(all_labels)
        ref_ids = all_ids
        
        # Save reference embeddings for future use
        ref_save_path = output_dir / 'reference_embeddings.npz'
        np.savez(
            ref_save_path,
            embeddings=ref_embeddings,
            labels=ref_labels,
            ids=ref_ids
        )
        logger.info(f"Reference embeddings saved to {ref_save_path}")
    
    # Create predictor
    logger.info("Initializing predictor...")
    predictor = NitroplastPredictor(
        model=model,
        reference_embeddings=ref_embeddings,
        reference_labels=ref_labels,
        reference_ids=ref_ids,
        distance_metric=config['inference']['distance_metric'],
        confidence_threshold=config['inference']['confidence_threshold'],
        device=device
    )
    
    # Predict on full proteome
    logger.info(f"Predicting localizations for {args.proteome}")
    predictions_df = predictor.predict_from_fasta(
        fasta_path=args.proteome,
        output_path=str(output_dir / 'novel_candidates.csv'),
        min_confidence=config['inference']['min_prediction_score'],
        return_top_k=config['inference']['return_top_k']
    )
    
    logger.info(f"\nFound {len(predictions_df)} novel nitroplast candidates")
    logger.info(f"Top 5 predictions:")
    print(predictions_df[['protein_id', 'confidence', 'distance_to_positive']].head())
    
    # Optionally visualize
    if args.visualize:
        logger.info("\nGenerating visualizations...")
        
        # Load all proteome sequences
        proteome_sequences, proteome_ids = load_fasta(args.proteome)
        
        # Compute embeddings for all proteins
        logger.info("Computing embeddings for full proteome...")
        all_proteome_embeddings = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(proteome_sequences), args.batch_size):
                batch_seqs = proteome_sequences[i:i+args.batch_size]
                embeddings = model.get_embeddings(batch_seqs).cpu().numpy()
                all_proteome_embeddings.append(embeddings)
        
        proteome_embeddings = np.vstack(all_proteome_embeddings)
        
        # Get predictions for all
        proteome_predictions = predictor.predict_batch(
            sequences=proteome_sequences,
            protein_ids=proteome_ids,
            batch_size=args.batch_size
        )
        proteome_labels = np.array([p['prediction'] for p in proteome_predictions])
        
        # Combine with reference data
        combined_embeddings = np.vstack([ref_embeddings, proteome_embeddings])
        combined_labels = np.concatenate([
            ref_labels,
            proteome_labels
        ])
        combined_ids = ref_ids + proteome_ids
        
        # Create metadata to distinguish known vs predicted
        metadata = {}
        for i, protein_id in enumerate(combined_ids):
            if i < len(ref_ids):
                metadata[protein_id] = {'source': 'known'}
            else:
                metadata[protein_id] = {'source': 'predicted'}
        
        # Visualize
        visualize_embeddings(
            embeddings=combined_embeddings,
            labels=combined_labels,
            ids=combined_ids,
            output_path=str(output_dir / 'proteome_predictions_tsne.png'),
            method='tsne',
            title='Full Proteome Predictions (t-SNE)',
            metadata=metadata
        )
        
        logger.info(f"Visualization saved to {output_dir / 'proteome_predictions_tsne.png'}")
    
    # Cluster novel predictions
    if args.cluster and len(predictions_df) >= config['inference']['dbscan_min_samples']:
        logger.info("\nClustering novel predictions...")
        
        # Get embeddings for predicted proteins
        predicted_ids = predictions_df['protein_id'].tolist()
        predicted_indices = [proteome_ids.index(pid) for pid in predicted_ids]
        predicted_embeddings = proteome_embeddings[predicted_indices]
        predicted_labels = np.ones(len(predicted_indices))
        
        # Cluster
        cluster_labels = predictor.cluster_predictions(
            embeddings=predicted_embeddings,
            labels=predicted_labels,
            eps=config['inference']['dbscan_eps'],
            min_samples=config['inference']['dbscan_min_samples']
        )
        
        # Add cluster assignments to predictions
        predictions_df['cluster'] = cluster_labels
        predictions_df.to_csv(output_dir / 'novel_candidates_clustered.csv', index=False)
        
        # Print cluster statistics
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
            cluster_size = (cluster_labels == cluster_id).sum()
            logger.info(f"Cluster {cluster_id}: {cluster_size} proteins")
    
    logger.info(f"\n✓ Prediction complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review novel candidates in: {output_dir / 'novel_candidates.csv'}")
    logger.info(f"2. Analyze attention maps for hidden signals (use visualization script)")
    logger.info(f"3. Validate predictions experimentally")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict nitroplast localization for novel proteins"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--proteome',
        type=str,
        required=True,
        help='Path to full proteome FASTA file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--reference-embeddings',
        type=str,
        default=None,
        help='Path to precomputed reference embeddings (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate t-SNE visualization of predictions'
    )
    parser.add_argument(
        '--cluster',
        action='store_true',
        help='Cluster novel predictions using DBSCAN'
    )
    
    args = parser.parse_args()
    main(args)
