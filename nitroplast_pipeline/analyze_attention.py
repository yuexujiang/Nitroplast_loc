"""
Attention analysis script for identifying novel targeting signals.

Analyzes the 148 "hidden" proteins (nitroplast-localized without uTP motif)
to discover what features the model is attending to.

Usage:
    python analyze_attention.py --checkpoint results/checkpoints/best_model.pt \
                                 --target-proteins data/raw/hidden_148.fasta \
                                 --output results/attention_maps/
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from utils.inference_utils import load_model_for_inference
from utils.data_utils import load_fasta
from utils.visualization import (
    analyze_attention_batch,
    extract_high_attention_regions
)
from Bio import SeqIO
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_motif_position(high_attention_regions: pd.DataFrame, sequence_lengths: dict):
    """
    Analyze where high-attention regions are located in sequences.
    
    Args:
        high_attention_regions: DataFrame with high-attention regions
        sequence_lengths: Dict mapping protein_id -> sequence length
    """
    logger.info("\n=== Motif Position Analysis ===")
    
    # Categorize regions by position
    n_terminal = 0  # First 50 residues
    c_terminal = 0  # Last 50 residues
    middle = 0
    
    for _, row in high_attention_regions.iterrows():
        protein_id = row['protein_id']
        start = row['start']
        seq_len = sequence_lengths[protein_id]
        
        if start < 50:
            n_terminal += 1
        elif start > seq_len - 50:
            c_terminal += 1
        else:
            middle += 1
    
    total = len(high_attention_regions)
    logger.info(f"N-terminal (first 50 aa): {n_terminal} ({n_terminal/total*100:.1f}%)")
    logger.info(f"C-terminal (last 50 aa): {c_terminal} ({c_terminal/total*100:.1f}%)")
    logger.info(f"Middle region: {middle} ({middle/total*100:.1f}%)")


def analyze_amino_acid_composition(high_attention_regions: pd.DataFrame):
    """
    Analyze amino acid composition in high-attention regions.
    """
    logger.info("\n=== Amino Acid Composition Analysis ===")
    
    # Concatenate all high-attention sequences
    all_sequences = ''.join(high_attention_regions['sequence'].tolist())
    
    # Count amino acids
    aa_counts = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        count = all_sequences.count(aa)
        aa_counts[aa] = count
    
    total = sum(aa_counts.values())
    
    # Sort by frequency
    sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Top 10 amino acids in high-attention regions:")
    for aa, count in sorted_aa[:10]:
        freq = count / total * 100
        logger.info(f"  {aa}: {count} ({freq:.2f}%)")
    
    # Analyze biophysical properties
    charged = sum(aa_counts[aa] for aa in 'DEKR')
    hydrophobic = sum(aa_counts[aa] for aa in 'AILMFWV')
    polar = sum(aa_counts[aa] for aa in 'STNQCY')
    
    logger.info("\nBiophysical properties:")
    logger.info(f"  Charged (D,E,K,R): {charged/total*100:.1f}%")
    logger.info(f"  Hydrophobic (A,I,L,M,F,W,V): {hydrophobic/total*100:.1f}%")
    logger.info(f"  Polar (S,T,N,Q,C,Y): {polar/total*100:.1f}%")


def find_consensus_motifs(high_attention_regions: pd.DataFrame, min_support: int = 5):
    """
    Look for conserved motifs in high-attention regions.
    
    Uses simple k-mer counting to find shared subsequences.
    """
    logger.info("\n=== Consensus Motif Discovery ===")
    
    sequences = high_attention_regions['sequence'].tolist()
    
    # Count k-mers for various lengths
    for k in [3, 4, 5]:
        kmer_counts = {}
        
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
        
        # Find frequently occurring k-mers
        frequent = [(kmer, count) for kmer, count in kmer_counts.items() 
                   if count >= min_support]
        frequent.sort(key=lambda x: x[1], reverse=True)
        
        if frequent:
            logger.info(f"\nTop {k}-mers (appearing in ≥{min_support} proteins):")
            for kmer, count in frequent[:10]:
                logger.info(f"  {kmer}: {count} occurrences")


def compare_with_known_utp(
    model,
    hidden_proteins: list,
    utp_proteins: list,
    device: str = "cuda"
):
    """
    Compare attention patterns between uTP+ and uTP- proteins.
    """
    logger.info("\n=== Comparison with Known uTP Motif ===")
    
    # Get attention for both groups
    hidden_seqs, hidden_ids = zip(*hidden_proteins)
    utp_seqs, utp_ids = zip(*utp_proteins)
    
    model.eval()
    with torch.no_grad():
        # Hidden proteins (uTP-)
        hidden_attention_list = []
        for seq in hidden_seqs:
            attention = model.encoder.get_attention_weights(
                [seq], layer_idx=-1, aggregate_heads="mean"
            )
            attention_scores = attention[0].cpu().numpy().mean(axis=0)
            hidden_attention_list.append(attention_scores)
        
        # uTP+ proteins
        utp_attention_list = []
        for seq in utp_seqs[:len(hidden_seqs)]:  # Match sample size
            attention = model.encoder.get_attention_weights(
                [seq], layer_idx=-1, aggregate_heads="mean"
            )
            attention_scores = attention[0].cpu().numpy().mean(axis=0)
            utp_attention_list.append(attention_scores)
    
    # Analyze attention distribution along sequence
    logger.info("\nAttention distribution:")
    
    # Divide sequence into thirds
    for group_name, attention_list, seqs in [
        ("Hidden (uTP-)", hidden_attention_list, hidden_seqs),
        ("Known (uTP+)", utp_attention_list, utp_seqs[:len(hidden_seqs)])
    ]:
        n_terminal_attn = []
        middle_attn = []
        c_terminal_attn = []
        
        for attention, seq in zip(attention_list, seqs):
            seq_len = len(seq)
            third = seq_len // 3
            
            n_terminal_attn.append(attention[:third].mean())
            middle_attn.append(attention[third:2*third].mean())
            c_terminal_attn.append(attention[2*third:].mean())
        
        logger.info(f"\n{group_name}:")
        logger.info(f"  N-terminal: {np.mean(n_terminal_attn):.4f} ± {np.std(n_terminal_attn):.4f}")
        logger.info(f"  Middle: {np.mean(middle_attn):.4f} ± {np.std(middle_attn):.4f}")
        logger.info(f"  C-terminal: {np.mean(c_terminal_attn):.4f} ± {np.std(c_terminal_attn):.4f}")


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device
    )
    
    # Load target proteins (the 148 hidden proteins)
    logger.info(f"Loading target proteins from {args.target_proteins}")
    sequences, ids = load_fasta(args.target_proteins)
    sequence_lengths = {pid: len(seq) for pid, seq in zip(ids, sequences)}
    
    logger.info(f"Loaded {len(sequences)} proteins for analysis")
    
    # Batch analyze attention
    logger.info("\nAnalyzing attention patterns...")
    analyze_attention_batch(
        model=model,
        sequences=sequences,
        ids=ids,
        output_dir=str(output_dir),
        layer_idx=args.layer,
        aggregate_heads=args.aggregate_heads,
        top_k=args.top_k,
        window_size=args.window_size
    )
    
    # Load high-attention regions
    regions_path = output_dir / "high_attention_regions.csv"
    high_attention_regions = pd.read_csv(regions_path)
    
    # Analyze position distribution
    analyze_motif_position(high_attention_regions, sequence_lengths)
    
    # Analyze amino acid composition
    analyze_amino_acid_composition(high_attention_regions)
    
    # Find consensus motifs
    find_consensus_motifs(high_attention_regions, min_support=args.min_support)
    
    # Compare with known uTP proteins (if provided)
    if args.utp_proteins:
        logger.info(f"\nLoading uTP+ proteins from {args.utp_proteins}")
        utp_seqs, utp_ids = load_fasta(args.utp_proteins)
        
        compare_with_known_utp(
            model=model,
            hidden_proteins=list(zip(sequences, ids)),
            utp_proteins=list(zip(utp_seqs, utp_ids)),
            device=device
        )
    
    # Generate summary report
    summary_path = output_dir / "attention_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NITROPLAST TARGETING SIGNAL DISCOVERY - ATTENTION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analyzed {len(sequences)} proteins without known uTP motif\n")
        f.write(f"Identified {len(high_attention_regions)} high-attention regions\n\n")
        f.write("Key Findings:\n")
        f.write("1. Individual attention maps saved for each protein\n")
        f.write("2. High-attention regions extracted and saved to CSV\n")
        f.write("3. Motif position, amino acid composition, and consensus patterns analyzed\n\n")
        f.write(f"See detailed results in: {output_dir}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"\n✓ Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary report: {summary_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review attention maps for individual proteins")
    logger.info(f"2. Examine high-attention regions in {regions_path}")
    logger.info(f"3. Look for shared patterns or novel motifs")
    logger.info(f"4. Design experiments to validate discovered signals")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns to discover novel targeting signals"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--target-proteins',
        type=str,
        required=True,
        help='Path to FASTA file with proteins to analyze (148 hidden proteins)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for attention maps and analysis'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--utp-proteins',
        type=str,
        default=None,
        help='Optional: FASTA file with uTP+ proteins for comparison'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=-1,
        help='Which transformer layer to visualize (-1 for last layer)'
    )
    parser.add_argument(
        '--aggregate-heads',
        type=str,
        default='mean',
        choices=['mean', 'max', None],
        help='How to aggregate attention heads'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top-attended residues to highlight'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=15,
        help='Window size for extracting high-attention regions'
    )
    parser.add_argument(
        '--min-support',
        type=int,
        default=5,
        help='Minimum number of proteins for consensus motif'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    
    args = parser.parse_args()
    main(args)
