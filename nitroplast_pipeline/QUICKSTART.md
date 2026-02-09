# Quick Start Guide: Nitroplast Import Code Discovery

## Prerequisites
```bash
# Create conda environment
conda create -n nitroplast python=3.9
conda activate nitroplast

# Install dependencies
pip install -r requirements.txt

# Install MMseqs2 for sequence clustering (optional but recommended)
conda install -c bioconda mmseqs2
```

## Directory Setup

Place your data files in the following locations:
```
data/
├── raw/
│   ├── nitroplast_proteins.fasta    # 368 positive proteins
│   ├── host_cytosolic.fasta         # 788 negative proteins
│   ├── hidden_148.fasta             # 148 proteins without uTP motif
│   └── full_proteome.fasta          # Complete B. bigelowii proteome
```

## Workflow

### Step 1: Data Preparation
```bash
python -m utils.data_utils --config configs/config.yaml
```

This will:
- Load and validate sequences
- Cluster by similarity (70% identity threshold)
- Split into train/val/test sets
- Save processed data to `data/processed/`

### Step 2: Train Model
```bash
python train.py --config configs/config.yaml
```

Expected runtime: 4-8 hours on a V100 GPU (100 epochs)

Monitor training:
```bash
tensorboard --logdir results/tensorboard/
```

### Step 3: Analyze Attention (Discover Hidden Signals)
```bash
python analyze_attention.py \
    --checkpoint results/checkpoints/best_model.pt \
    --target-proteins data/raw/hidden_148.fasta \
    --output results/attention_maps/
```

This will generate:
- Individual attention maps for each protein
- High-attention regions CSV
- Amino acid composition analysis
- Consensus motif discovery

### Step 4: Zero-Shot Prediction
```bash
python predict.py \
    --checkpoint results/checkpoints/best_model.pt \
    --proteome data/raw/full_proteome.fasta \
    --output results/predictions/ \
    --visualize \
    --cluster
```

This will:
- Predict nitroplast localization for all proteins
- Return top candidates above confidence threshold
- Cluster predictions to find groups with similar signals
- Generate t-SNE visualization

## Configuration

Edit `configs/config.yaml` to customize:

### Key hyperparameters:
- `model.lora_r`: LoRA rank (default: 8)
- `model.freeze_esm_layers`: Number of ESM layers to freeze (default: 20)
- `training.temperature`: Contrastive temperature (default: 0.07)
- `training.learning_rate`: Learning rate (default: 2e-4)
- `training.batch_size`: Batch size (default: 16)

### For faster training:
```yaml
model:
  esm_model_name: "facebook/esm2_t30_150M_UR50D"  # Use smaller model
  freeze_esm_layers: 25  # Freeze more layers
training:
  batch_size: 32  # Larger batches
  mixed_precision: true  # Enable AMP
```

### For better accuracy:
```yaml
model:
  lora_r: 16  # Higher rank
  freeze_esm_layers: 15  # Fine-tune more layers
training:
  temperature: 0.05  # Lower temperature (sharper contrasts)
  num_epochs: 150  # More epochs
```

## Expected Results

### Training metrics (final):
- Val Loss: ~0.15-0.25
- Val Silhouette Score: >0.5
- Val Davies-Bouldin Index: <1.0

### Attention analysis:
- Should identify high-attention regions in uTP- proteins
- Look for patterns in position (N-/C-terminal vs middle)
- Check amino acid composition (charged, hydrophobic, etc.)

### Zero-shot predictions:
- Typical: 50-100 novel candidates above confidence threshold
- Cluster analysis may reveal 3-5 distinct signal groups

## Troubleshooting

### Out of memory error:
```bash
# Reduce batch size
python train.py --config configs/config.yaml
# Edit config.yaml: training.batch_size: 8

# Or use gradient accumulation
training:
  batch_size: 8
  gradient_accumulation_steps: 2  # Effective batch = 16
```

### MMseqs2 not found:
The pipeline will fall back to identity-based splitting if MMseqs2 is unavailable, but clustering is recommended for better generalization.

### Slow training:
- Enable mixed precision: `training.mixed_precision: true`
- Use smaller ESM model: `esm2_t30_150M_UR50D`
- Reduce validation frequency

## Output Files

```
results/
├── checkpoints/
│   └── best_model.pt              # Trained model
├── embeddings/
│   └── test_embeddings.npz        # Protein embeddings
├── plots/
│   ├── training_curves.png        # Loss and metrics
│   └── test_embeddings_tsne.png   # Embedding visualization
├── attention_maps/
│   ├── PROTEIN_ID_attention.png   # Individual attention maps
│   └── high_attention_regions.csv # Extracted regions
└── predictions/
    ├── novel_candidates.csv       # Zero-shot predictions
    └── reference_embeddings.npz   # Cached embeddings
```

## Semi-Supervised Approach Explained

**Question**: Why treat some negatives as "unlabeled"?

**Answer**: Your 788 "cytosolic" proteins are based on proteomics, which isn't perfect. Some might actually be nitroplast-localized but just weren't captured.

To use semi-supervised learning:

1. **Filter confident negatives** (edit `configs/config.yaml`):
```yaml
data:
  use_high_confidence_negatives_only: true
  confident_negative_criteria:
    exclude_membrane_proteins: true
    exclude_secreted_proteins: true
```

2. **Future extension**: Use remaining proteins as unlabeled pool for pseudo-labeling or consistency regularization.

For now, the pipeline treats all negatives as confident, but you can implement filtering based on your domain knowledge.

## Next Steps for Your Research

1. **Validate attention findings**:
   - Do high-attention regions show conservation across homologs?
   - Are they enriched in secondary structures (helices, disorder)?

2. **Experimental validation**:
   - GFP-fusion localization assays for top predictions
   - Mutagenesis of high-attention regions

3. **Comparative analysis**:
   - Compare with other organellar targeting signals (mitochondria, chloroplast)
   - Look for evolutionary conservation in UCYN-A relatives

4. **Publication**:
   - Methods: ESM-2 + LoRA + SupCon
   - Results: Novel targeting signal discovery + proteome predictions
   - Impact: First systematic analysis of nitroplast import machinery

## Citation

If you use this pipeline, please cite:
- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science 2023
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- **SupCon**: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020

## Support

For questions or issues:
1. Check the comprehensive tutorial: `notebooks/tutorial.ipynb`
2. Review the README: `README.md`
3. Examine example outputs in `results/`
