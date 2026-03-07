"""
5-fold cross-validation training for nitroplast localization model.

For each fold:
  - Split is done at the CLUSTER level to prevent homology leakage.
  - The 4 non-test folds are further split 85/15 into train / val (early stopping).
  - After training, the best checkpoint is evaluated on:
      1. The full test fold (test_dataset)
      2. A sub-test dataset: test proteins whose IDs appear in
         positive_no_uTP.fasta OR uTP_in_negative.fasta (same logic as
         tutorial.ipynb section 11).
  - Classification metrics (accuracy, recall, F1, MCC, confusion matrix)
    are computed for both subsets using a nearest-centroid predictor
    built from the training embeddings.
  - A log file is saved to <output_dir>/cv_training.log
  - Per-fold JSON results and a cross-validation summary are saved to
    <output_dir>/cv_results/.

Usage:
    python train_cv.py --config configs/config.yaml [--n-folds 5]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from models.esm_encoder import ESMEncoder
from models.projector import NitroplastContrastiveModel, ProjectionHead, SupConLoss
from utils.data_utils import (
    ProteinSequenceDataset,
    cluster_sequences_mmseqs,
    collate_fn,
    load_fasta,
    validate_sequences,
)
from utils.inference_utils import NitroplastPredictor, compute_reference_embeddings
from utils.training_utils import (
    EarlyStopping,
    MetricsTracker,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    train_epoch,
    validate,
)


# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    """Configure root logger with both console and file handlers."""
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger("train_cv")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Dataset helpers ───────────────────────────────────────────────────────────
def make_dataset(all_sequences, all_labels, all_ids, indices):
    return ProteinSequenceDataset(
        sequences=[all_sequences[i] for i in indices],
        labels=[all_labels[i] for i in indices],
        ids=[all_ids[i] for i in indices],
    )


def filter_sub_test(test_dataset: ProteinSequenceDataset, id_set: set):
    """Return a subset of test_dataset whose IDs are in id_set."""
    indices = [i for i, pid in enumerate(test_dataset.ids) if pid in id_set]
    if not indices:
        return None
    return ProteinSequenceDataset(
        sequences=[test_dataset.sequences[i] for i in indices],
        labels=[test_dataset.labels[i] for i in indices],
        ids=[test_dataset.ids[i] for i in indices],
    )


# ── Cross-validation splitter ─────────────────────────────────────────────────
def make_cv_splits(clusters, all_labels, n_folds=5, val_fraction=0.15, seed=42):
    """
    Produce n_folds cross-validation splits at the CLUSTER level.

    Each fold dict contains keys:
        'fold'  : 1-indexed fold number
        'train' : list of sequence indices for training
        'val'   : list of sequence indices for validation (early stopping)
        'test'  : list of sequence indices for held-out test
    """
    cluster_ids = list(clusters.keys())
    cluster_labels = np.array(
        [
            1 if sum(all_labels[idx] for idx in clusters[cid]) > len(clusters[cid]) / 2 else 0
            for cid in cluster_ids
        ]
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        skf.split(cluster_ids, cluster_labels)
    ):
        # Test sequences
        test_seq_indices = []
        for ci in test_idx:
            test_seq_indices.extend(clusters[cluster_ids[ci]])

        # Split train+val clusters into train / val
        tv_cluster_ids = [cluster_ids[i] for i in trainval_idx]
        tv_labels = cluster_labels[trainval_idx]

        # Stratified split of clusters if both classes are present
        if len(set(tv_labels)) > 1:
            train_ci, val_ci = train_test_split(
                range(len(tv_cluster_ids)),
                test_size=val_fraction,
                stratify=tv_labels,
                random_state=seed,
            )
        else:
            n_val = max(1, int(len(tv_cluster_ids) * val_fraction))
            val_ci = list(range(n_val))
            train_ci = list(range(n_val, len(tv_cluster_ids)))

        train_seq_indices = []
        for ci in train_ci:
            train_seq_indices.extend(clusters[tv_cluster_ids[ci]])

        val_seq_indices = []
        for ci in val_ci:
            val_seq_indices.extend(clusters[tv_cluster_ids[ci]])

        fold_splits.append(
            {
                "fold": fold_idx + 1,
                "train": train_seq_indices,
                "val": val_seq_indices,
                "test": test_seq_indices,
            }
        )

    return fold_splits


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(config, device):
    encoder = ESMEncoder(
        model_name=config["model"]["esm_model_name"],
        use_lora=config["model"]["use_lora"],
        lora_config={
            "r": config["model"]["lora_r"],
            "alpha": config["model"]["lora_alpha"],
            "dropout": config["model"]["lora_dropout"],
            "target_modules": config["model"]["lora_target_modules"],
        },
        num_end_lora_layers=config["model"].get("esm_num_end_lora"),
        freeze_layers=config["model"]["freeze_esm_layers"],
        pooling_method=config["model"]["pooling_method"],
        device=device,
    )
    projector = ProjectionHead(
        input_dim=config["model"]["projector"]["input_dim"],
        hidden_dims=config["model"]["projector"]["hidden_dims"],
        output_dim=config["model"]["projector"]["output_dim"],
        dropout=config["model"]["projector"]["dropout"],
        use_batch_norm=config["model"]["projector"]["use_batch_norm"],
    )
    return NitroplastContrastiveModel(encoder, projector).to(device)


# ── Classification metrics ────────────────────────────────────────────────────
def compute_classification_metrics(y_true, y_pred, logger, tag=""):
    """Log and return a dict of classification metrics."""
    if len(y_true) == 0:
        logger.warning(f"[{tag}] Empty dataset — skipping metrics.")
        return None

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    logger.info(f"  [{tag}]  n={len(y_true)}  pos={int(sum(y_true))}  neg={len(y_true)-int(sum(y_true))}")
    logger.info(f"  [{tag}]  Accuracy : {acc:.4f}")
    logger.info(f"  [{tag}]  Recall   : {rec:.4f}")
    logger.info(f"  [{tag}]  F1       : {f1:.4f}")
    logger.info(f"  [{tag}]  MCC      : {mcc:.4f}")
    logger.info(f"  [{tag}]  Confusion matrix (rows=true, cols=pred):")
    logger.info(f"             Pred-Neg  Pred-Pos")
    logger.info(f"  True-Neg   {cm[0][0]:>6}    {cm[0][1]:>6}")
    logger.info(f"  True-Pos   {cm[1][0]:>6}    {cm[1][1]:>6}")
    logger.info(
        "\n"
        + classification_report(
            y_true, y_pred, target_names=["Cytosolic", "Nitroplast"], zero_division=0
        )
    )

    return {
        "accuracy": float(acc),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "confusion_matrix": cm,
        "n_samples": len(y_true),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
    }


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_with_predictor(model, train_dataset, eval_dataset, config, device, logger, tag):
    """
    Build a nearest-centroid predictor from train embeddings and evaluate
    on eval_dataset.  Returns a metrics dict (or None if eval_dataset is empty).
    """
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.info(f"  [{tag}] Dataset is empty — skipping evaluation.")
        return None

    # Build reference from training set
    ref_emb, ref_labels, ref_ids = compute_reference_embeddings(
        model=model,
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    predictor = NitroplastPredictor(
        model=model,
        reference_embeddings=ref_emb,
        reference_labels=ref_labels,
        reference_ids=ref_ids,
        distance_metric=config["inference"]["distance_metric"],
        device=device,
    )

    predictions = predictor.predict_batch(
        sequences=eval_dataset.sequences,
        protein_ids=eval_dataset.ids,
        batch_size=config["training"]["batch_size"],
    )

    y_true = np.array(eval_dataset.labels)
    y_pred = np.array([p["prediction"] for p in predictions])

    return compute_classification_metrics(y_true, y_pred, logger, tag=tag)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    n_folds = args.n_folds
    seed = config["seed"]
    set_seed(seed)

    # ── Output directories ────────────────────────────────────────────────────
    output_dir = Path(config["paths"]["output_dir"])
    cv_dir = output_dir / "cv_results"
    cv_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logging(output_dir / "cv_training.log")
    logger.info(f"Starting {n_folds}-fold cross-validation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {output_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── Load all sequences ────────────────────────────────────────────────────
    logger.info("Loading sequences from FASTA files...")
    pos_sequences, pos_ids = load_fasta(config["paths"]["positive_fasta"])
    neg_sequences, neg_ids = load_fasta(config["paths"]["negative_fasta"])

    all_sequences = pos_sequences + neg_sequences
    all_ids = pos_ids + neg_ids
    all_labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)

    # ── Validate ──────────────────────────────────────────────────────────────
    logger.info("Validating sequences...")
    all_sequences, valid_indices = validate_sequences(
        all_sequences,
        min_length=config["data"]["min_sequence_length"],
        max_length=config["data"]["max_sequence_length"],
    )
    all_ids = [all_ids[i] for i in valid_indices]
    all_labels = [all_labels[i] for i in valid_indices]
    logger.info(f"Total valid sequences: {len(all_sequences)}")

    # ── Cluster ───────────────────────────────────────────────────────────────
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Clustering sequences by similarity (MMseqs2)...")
    clusters = cluster_sequences_mmseqs(
        all_sequences,
        all_ids,
        similarity_threshold=config["data"]["sequence_similarity_threshold"],
        output_dir=str(processed_dir / "clustering_cv"),
        force_recompute=args.force_recompute,
    )
    logger.info(f"Total clusters: {len(clusters)}")

    # ── Sub-test ID set (from notebook section 11) ────────────────────────────
    logger.info("Loading sub-test FASTA files...")
    sub_test_id_set = set()

    no_utp_fasta = Path(config["paths"].get("no_utp_fasta", "data/raw/positive_no_uTP.fasta"))
    utp_neg_fasta = Path(config["paths"].get("utp_neg_fasta", "data/raw/uTP_in_negative.fasta"))

    for fasta_path in [no_utp_fasta, utp_neg_fasta]:
        if fasta_path.exists():
            _, ids = load_fasta(str(fasta_path))
            sub_test_id_set.update(ids)
            logger.info(f"  Loaded {len(ids)} IDs from {fasta_path}")
        else:
            logger.warning(f"  Sub-test FASTA not found: {fasta_path}")

    logger.info(f"Total sub-test ID set size: {len(sub_test_id_set)}")

    # ── Build CV splits ───────────────────────────────────────────────────────
    logger.info(f"Building {n_folds}-fold CV splits at cluster level...")
    fold_splits = make_cv_splits(clusters, all_labels, n_folds=n_folds, seed=seed)

    for fs in fold_splits:
        train_labels = [all_labels[i] for i in fs["train"]]
        val_labels = [all_labels[i] for i in fs["val"]]
        test_labels = [all_labels[i] for i in fs["test"]]
        logger.info(
            f"  Fold {fs['fold']}: "
            f"train={len(fs['train'])} (pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)}), "
            f"val={len(fs['val'])} (pos={sum(val_labels)}, neg={len(val_labels)-sum(val_labels)}), "
            f"test={len(fs['test'])} (pos={sum(test_labels)}, neg={len(test_labels)-sum(test_labels)})"
        )

    # ── Loss function ─────────────────────────────────────────────────────────
    loss_fn = SupConLoss(temperature=config["training"]["temperature"])

    # ── Per-fold results accumulator ──────────────────────────────────────────
    all_fold_results = []

    # ═══════════════════════════════════════════════════════════════════════════
    for fs in fold_splits:
        fold = fs["fold"]
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  FOLD {fold}/{n_folds}")
        logger.info("=" * 70)

        set_seed(seed + fold)  # different seed per fold, but reproducible

        # ── Datasets & loaders ────────────────────────────────────────────────
        train_dataset = make_dataset(all_sequences, all_labels, all_ids, fs["train"])
        val_dataset = make_dataset(all_sequences, all_labels, all_ids, fs["val"])
        test_dataset = make_dataset(all_sequences, all_labels, all_ids, fs["test"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
        )

        # ── Model ─────────────────────────────────────────────────────────────
        logger.info(f"[Fold {fold}] Building fresh model...")
        model = build_model(config, device)

        # ── Optimizer & scheduler ─────────────────────────────────────────────
        optimizer = create_optimizer(
            model,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        num_training_steps = len(train_loader) * config["training"]["num_epochs"]
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=config["training"]["warmup_steps"],
            scheduler_type=config["training"]["scheduler"],
            min_lr=config["training"]["min_lr"],
        )

        scaler = None
        if config["training"]["mixed_precision"] and device == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        # ── Checkpoint dir for this fold ─────────────────────────────────────
        fold_ckpt_dir = Path(config["paths"]["checkpoint_dir"]) / f"fold_{fold}"
        fold_ckpt_dir.mkdir(exist_ok=True, parents=True)

        # ── Early stopping & metrics ──────────────────────────────────────────
        early_stopping = EarlyStopping(
            patience=config["training"]["patience"],
            min_delta=config["training"]["min_delta"],
            mode="min",
        )
        metrics_tracker = MetricsTracker()
        best_val_loss = float("inf")

        # ── Training loop ─────────────────────────────────────────────────────
        logger.info(f"[Fold {fold}] Starting training...")
        for epoch in range(1, config["training"]["num_epochs"] + 1):
            train_loss, train_metrics = train_epoch(
                model=model,
                dataloader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                grad_clip=config["training"]["gradient_clip_norm"],
                scaler=scaler,
            )
            val_loss, val_metrics = validate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )

            if scheduler is not None:
                scheduler.step()

            logger.info(
                f"[Fold {fold}] Epoch {epoch:>3}/{config['training']['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Sil: {train_metrics['silhouette']:.4f} | "
                f"Val Sil: {val_metrics['silhouette']:.4f}"
            )

            metrics_tracker.update(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_silhouette": train_metrics["silhouette"],
                    "val_silhouette": val_metrics["silhouette"],
                    "train_davies_bouldin": train_metrics["davies_bouldin"],
                    "val_davies_bouldin": val_metrics["davies_bouldin"],
                }
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    save_path=str(fold_ckpt_dir / "best_model.pt"),
                )
                logger.info(f"[Fold {fold}]   -> New best model (val_loss={val_loss:.4f})")

            # Periodic checkpoint
            if epoch % config["training"]["save_every_n_epochs"] == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    save_path=str(fold_ckpt_dir / f"checkpoint_epoch_{epoch}.pt"),
                )

            if early_stopping(val_loss):
                logger.info(f"[Fold {fold}] Early stopping at epoch {epoch}")
                break

        # Save training metrics for this fold
        metrics_tracker.save(str(cv_dir / f"fold_{fold}_training_metrics.json"))

        # ── Load best model ───────────────────────────────────────────────────
        logger.info(f"[Fold {fold}] Loading best checkpoint...")
        ckpt = torch.load(fold_ckpt_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # ── Evaluate on test set ──────────────────────────────────────────────
        logger.info(f"[Fold {fold}] Evaluating on test set...")
        test_metrics = evaluate_with_predictor(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            config=config,
            device=device,
            logger=logger,
            tag=f"Fold{fold}/test",
        )

        # ── Extract and evaluate sub-test set (section 11 logic) ─────────────
        sub_test_dataset = filter_sub_test(test_dataset, sub_test_id_set)
        if sub_test_dataset is not None:
            logger.info(
                f"[Fold {fold}] Sub-test set: {len(sub_test_dataset)} proteins "
                f"(pos={sum(sub_test_dataset.labels)}, "
                f"neg={len(sub_test_dataset)-sum(sub_test_dataset.labels)})"
            )
        else:
            logger.info(f"[Fold {fold}] Sub-test set: 0 proteins (none of the test proteins matched sub-test IDs)")

        sub_test_metrics = evaluate_with_predictor(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=sub_test_dataset,
            config=config,
            device=device,
            logger=logger,
            tag=f"Fold{fold}/sub_test",
        )

        # ── Collect fold results ──────────────────────────────────────────────
        fold_result = {
            "fold": fold,
            "best_val_loss": float(best_val_loss),
            "test_metrics": test_metrics,
            "sub_test_metrics": sub_test_metrics,
        }
        all_fold_results.append(fold_result)

        # Save per-fold results JSON
        with open(cv_dir / f"fold_{fold}_results.json", "w") as f:
            json.dump(fold_result, f, indent=2)
        logger.info(f"[Fold {fold}] Results saved to {cv_dir / f'fold_{fold}_results.json'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cross-validation summary
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 70)
    logger.info("  CROSS-VALIDATION SUMMARY")
    logger.info("=" * 70)

    def aggregate_metric(results, key, sub_key):
        vals = [
            r[key][sub_key]
            for r in results
            if r[key] is not None and sub_key in r[key]
        ]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"n_folds": n_folds, "folds": all_fold_results, "aggregate": {}}

    for subset_key, subset_label in [("test_metrics", "Test"), ("sub_test_metrics", "Sub-Test")]:
        logger.info(f"\n  {subset_label} Set Metrics:")
        subset_agg = {}
        for metric in ["accuracy", "recall", "f1", "mcc"]:
            mean, std = aggregate_metric(all_fold_results, subset_key, metric)
            if mean is not None:
                logger.info(f"    {metric:<12}: {mean:.4f} ± {std:.4f}")
                subset_agg[metric] = {"mean": mean, "std": std}
            else:
                logger.info(f"    {metric:<12}: N/A (no samples in any fold)")
        summary["aggregate"][subset_label.lower().replace("-", "_")] = subset_agg

    # Save overall summary
    summary_path = cv_dir / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nCross-validation summary saved to {summary_path}")
    logger.info("Training complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5-fold cross-validation training for nitroplast localization model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute of sequence clustering (ignore cache)",
    )
    args = parser.parse_args()
    main(args)
