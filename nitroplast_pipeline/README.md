# Nitroplast Import Code Discovery Pipeline

## Project Overview
Deep learning pipeline to discover hidden protein targeting signals for nitroplast import using ESM-2 embeddings and supervised contrastive learning.

## Directory Structure
```
nitroplast_pipeline/
├── data/
│   ├── raw/                    # Original FASTA files
│   ├── processed/              # Preprocessed data splits
│   └── embeddings/             # Cached ESM-2 embeddings
├── models/
│   ├── esm_encoder.py         # ESM-2 + LoRA wrapper
│   ├── projector.py           # Projection head for contrastive learning
│   └── contrastive_model.py   # Full model combining encoder + projector
├── utils/
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── training_utils.py      # Training loops and metrics
│   ├── visualization.py       # Attention maps and t-SNE plots
│   └── inference_utils.py     # Zero-shot prediction utilities
├── configs/
│   └── config.yaml            # Hyperparameters and paths
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_analysis.ipynb
├── results/
│   ├── checkpoints/           # Model weights
│   ├── embeddings/            # Final protein embeddings
│   ├── attention_maps/        # Visualizations
│   └── predictions/           # Zero-shot predictions
├── train.py                   # Main training script
├── predict.py                 # Inference script
└── requirements.txt           # Python dependencies
```

## Key Design Decisions

### 1. Pooling Strategy
- **Global average pooling** of all residue embeddings
- Rationale: uTP signals at C-terminus, unknown signals could be anywhere
- Alternative: Consider attention-weighted pooling for interpretability

### 2. Data Splitting Strategy
- **Sequence similarity-based splitting** (70% identity clusters)
- Ensures test proteins are evolutionarily distinct from training
- Prevents information leakage through homology

### 3. Semi-Supervised Approach
Instead of treating all 788 "host cytosolic" proteins as confident negatives:
- Use only high-confidence negatives (e.g., proteins with known cytosolic retention signals)
- Treat remainder as "unlabeled" pool for potential semi-supervised extensions

### 4. Contrastive Learning
- **Supervised Contrastive Loss (SupCon)** instead of SimCLR
- Pulls all positives together (both uTP+ and uTP-)
- Pushes away from negatives
- Temperature τ=0.07 (tunable)

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
```bash
python -m utils.data_utils \
    --positive_fasta data/raw/nitroplast_proteins.fasta \
    --negative_fasta data/raw/host_cytosolic.fasta \
    --output_dir data/processed/
```

### 2. Train Model
```bash
python train.py --config configs/config.yaml
```

### 3. Run Inference
```bash
python predict.py \
    --checkpoint results/checkpoints/best_model.pt \
    --proteome_fasta data/raw/full_proteome.fasta \
    --output results/predictions/
```

### 4. Visualize Attention
```bash
python -m utils.visualization \
    --checkpoint results/checkpoints/best_model.pt \
    --target_proteins data/raw/hidden_148_proteins.fasta \
    --output results/attention_maps/
```

## Expected Outputs
1. **Trained model**: `results/checkpoints/best_model.pt`
2. **Protein embeddings**: `results/embeddings/protein_embeddings.npz`
3. **Attention maps**: `results/attention_maps/protein_*.png`
4. **t-SNE visualization**: `results/embeddings/tsne_plot.png`
5. **Zero-shot predictions**: `results/predictions/novel_candidates.csv`

## Citation
If you use this pipeline, please cite:
- ESM-2: Lin et al., Science 2023
- LoRA: Hu et al., ICLR 2022
- SupCon: Khosla et al., NeurIPS 2020
