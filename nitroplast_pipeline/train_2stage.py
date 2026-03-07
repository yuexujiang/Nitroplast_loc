"""
2-stage training for nitroplast localization model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1 — Contrastive pretraining on ALL data (SupCon + LoRA)
  ALL positive and negative sequences are used to train the encoder.
  Because the goal of Stage 1 is to learn a general embedding space (not to
  produce a classifier), including every sequence maximises representation
  quality without affecting the classifier evaluation in Stage 2.
  A 10% held-out validation set (cluster-level, stratified) is used for
  early stopping on val contrastive loss to avoid collapse or overfitting.
  Best checkpoint (lowest val loss) is saved as stage1_best.pt.
  Checkpoint: <checkpoint_dir>/stage1_best.pt

Stage 2 — 5-fold CV MLP classifier (frozen encoder + projector)
  Freezes ALL Stage 1 weights.
  The full dataset is split into 5 folds at the CLUSTER level (same
  homology-aware logic as train_cv.py) to prevent leakage.
  For each fold:
    - Train split  (80 %): further divided 85/15 into train/val for
                           early stopping on val MCC.
    - Test split   (20 %): final evaluation of the fold.
    - Sub-test     : test proteins whose IDs appear in
                     positive_no_uTP.fasta OR uTP_in_negative.fasta.
  Classifier: 2-layer MLP  128 → BN → ReLU → Dropout → 64 → 2
  Loss       : class-weighted CrossEntropy (handles ~1:2 imbalance)
  Early stop : val MCC (more robust than loss for imbalanced data)

Log file: <output_dir>/stage_training.log
Results:  <output_dir>/stage2_cv_results/

Usage:
    python train_2stage.py --config configs/config.yaml
    python train_2stage.py --config configs/config.yaml --skip-stage1
    python train_2stage.py --config configs/config.yaml --n-folds 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import shutil


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils.training_utils import (
    EarlyStopping,
    MetricsTracker,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    train_epoch,
    validate,
)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger("train_2stage")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Model builder ─────────────────────────────────────────────────────────────
def build_contrastive_model(config, device):
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


# ── MLP classifier head ───────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    """
    2-layer MLP classifier trained on top of frozen contrastive embeddings.
    Architecture: Linear → BN → ReLU → Dropout → Linear
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Dataset helpers ───────────────────────────────────────────────────────────
def make_dataset(all_sequences, all_labels, all_ids, indices):
    return ProteinSequenceDataset(
        sequences=[all_sequences[i] for i in indices],
        labels=[all_labels[i] for i in indices],
        ids=[all_ids[i] for i in indices],
    )


def filter_sub_test(test_dataset, id_set):
    indices = [i for i, pid in enumerate(test_dataset.ids) if pid in id_set]
    if not indices:
        return None
    return ProteinSequenceDataset(
        sequences=[test_dataset.sequences[i] for i in indices],
        labels=[test_dataset.labels[i] for i in indices],
        ids=[test_dataset.ids[i] for i in indices],
    )


# ── Cluster-level CV splitter (same logic as train_cv.py) ─────────────────────
def make_cv_splits(clusters, all_labels, n_folds=5, val_fraction=0.15, seed=42):
    cluster_ids = list(clusters.keys())
    cluster_labels = np.array(
        [
            1 if sum(all_labels[idx] for idx in clusters[cid]) > len(clusters[cid]) / 2 else 0
            for cid in cluster_ids
        ]
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(cluster_ids, cluster_labels)):
        test_seq_indices = []
        for ci in test_idx:
            test_seq_indices.extend(clusters[cluster_ids[ci]])

        tv_cluster_ids = [cluster_ids[i] for i in trainval_idx]
        tv_labels = cluster_labels[trainval_idx]

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

        fold_splits.append({
            "fold": fold_idx + 1,
            "train": train_seq_indices,
            "val": val_seq_indices,
            "test": test_seq_indices,
        })

    return fold_splits


# ── Class-weight helper ───────────────────────────────────────────────────────
def compute_class_weights(labels, device):
    """Inverse-frequency weights: w_c = N / (2 * count_c)."""
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = len(labels) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Stage 2 train / eval loops ────────────────────────────────────────────────
def train_classifier_epoch(classifier, model, dataloader, optimizer, loss_fn, device):
    classifier.train()
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        with torch.no_grad():
            emb = model.get_embeddings(batch["sequences"])
        logits = classifier(emb)
        loss = loss_fn(logits, batch["labels"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_classifier(classifier, model, dataloader, loss_fn, device):
    classifier.eval()
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    for batch in dataloader:
        emb = model.get_embeddings(batch["sequences"])
        total_loss += loss_fn(classifier(emb), batch["labels"].to(device)).item()
        all_preds.extend(classifier(emb).argmax(dim=1).cpu().numpy())
        all_true.extend(batch["labels"].numpy())
    y_true, y_pred = np.array(all_true), np.array(all_preds)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return total_loss / len(dataloader), mcc, f1, y_true, y_pred


# ── Metrics logging ───────────────────────────────────────────────────────────
def log_metrics(y_true, y_pred, logger, tag):
    if y_true is None or len(y_true) == 0:
        logger.info(f"  [{tag}] Empty dataset — skipping.")
        return None
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    logger.info(f"  [{tag}]  n={len(y_true)}  pos={int(sum(y_true))}  neg={len(y_true)-int(sum(y_true))}")
    logger.info(f"  [{tag}]  Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    logger.info(f"  [{tag}]  Confusion matrix (rows=true, cols=pred):")
    logger.info(f"             Pred-Neg  Pred-Pos")
    logger.info(f"  True-Neg   {cm[0][0]:>6}    {cm[0][1]:>6}")
    logger.info(f"  True-Pos   {cm[1][0]:>6}    {cm[1][1]:>6}")
    logger.info("\n" + classification_report(
        y_true, y_pred, target_names=["Cytosolic", "Nitroplast"], zero_division=0))
    return {"accuracy": float(acc), "recall": float(rec), "f1": float(f1), "mcc": float(mcc),
            "confusion_matrix": cm, "n_samples": len(y_true),
            "n_positive": int(sum(y_true)), "n_negative": int(len(y_true) - sum(y_true))}


@torch.no_grad()
def predict_dataset(classifier, model, dataset, config, device):
    if dataset is None or len(dataset) == 0:
        return None, None
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=False, collate_fn=collate_fn)
    classifier.eval()
    model.eval()
    all_preds, all_true = [], []
    for batch in loader:
        emb = model.get_embeddings(batch["sequences"])
        all_preds.extend(classifier(emb).argmax(dim=1).cpu().numpy())
        all_true.extend(batch["labels"].numpy())
    return np.array(all_true), np.array(all_preds)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    # ── Dirs & logging ────────────────────────────────────────────────────────
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    cv_dir = output_dir / "stage2_cv_results"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    cv_dir.mkdir(exist_ok=True, parents=True)

    shutil.copy(args.config, output_dir)

    logger = setup_logging(output_dir / "stage_training.log")
    logger.info("2-Stage Training (Stage 1: SupCon + early stop on val loss | Stage 2: 5-fold CV MLP)")
    logger.info(f"Config: {args.config}")

    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── Load ALL sequences ────────────────────────────────────────────────────
    logger.info("Loading sequences...")
    pos_sequences, pos_ids = load_fasta(config["paths"]["positive_fasta"])
    neg_sequences, neg_ids = load_fasta(config["paths"]["negative_fasta"])

    all_sequences = pos_sequences + neg_sequences
    all_ids = pos_ids + neg_ids
    all_labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)

    logger.info("Validating sequences...")
    all_sequences, valid_indices = validate_sequences(
        all_sequences,
        min_length=config["data"]["min_sequence_length"],
        max_length=config["data"]["max_sequence_length"],
    )
    all_ids = [all_ids[i] for i in valid_indices]
    all_labels = [all_labels[i] for i in valid_indices]
    logger.info(f"Total sequences: {len(all_sequences)} "
                f"(pos={sum(all_labels)}, neg={len(all_labels)-sum(all_labels)})")

    # ── Cluster (for Stage 2 splits) ─────────────────────────────────────────
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Clustering sequences by similarity...")
    clusters = cluster_sequences_mmseqs(
        all_sequences, all_ids,
        similarity_threshold=config["data"]["sequence_similarity_threshold"],
        output_dir=str(processed_dir / "clustering_2stage"),
        force_recompute=args.force_recompute,
    )
    logger.info(f"Total clusters: {len(clusters)}")

    # ── Sub-test ID set ───────────────────────────────────────────────────────
    sub_test_id_set = set()
    for key, default in [("no_utp_fasta", "data/raw/positive_no_uTP.fasta"),
                         ("utp_neg_fasta", "data/raw/uTP_in_negative.fasta")]:
        p = Path(config["paths"].get(key, default))
        if p.exists():
            _, ids = load_fasta(str(p))
            sub_test_id_set.update(ids)
            logger.info(f"Sub-test: loaded {len(ids)} IDs from {p.name}")
        else:
            logger.warning(f"Sub-test FASTA not found: {p}")
    logger.info(f"Sub-test ID set: {len(sub_test_id_set)} total IDs")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Contrastive pretraining on ALL sequences
    # ═════════════════════════════════════════════════════════════════════════
    stage1_ckpt = checkpoint_dir / "stage1_best.pt"

    if args.skip_stage1 and stage1_ckpt.exists():
        logger.info(f"\nSkipping Stage 1 — loading: {stage1_ckpt}")
        model = build_contrastive_model(config, device)
        ckpt = torch.load(stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info("\n" + "=" * 65)
        logger.info("  STAGE 1 — Contrastive Pretraining on ALL data (SupCon + LoRA)")
        logger.info("=" * 65)

        # ── Stage 1: cluster-level 90/10 train/val split ─────────────────────
        logger.info("Creating Stage 1 train/val split (90% train, 10% val, cluster-level)...")
        s1_cluster_ids = list(clusters.keys())
        s1_cluster_labels = np.array(
            [
                1 if sum(all_labels[idx] for idx in clusters[cid]) > len(clusters[cid]) / 2 else 0
                for cid in s1_cluster_ids
            ]
        )
        s1_train_ci, s1_val_ci = train_test_split(
            range(len(s1_cluster_ids)),
            test_size=0.1,
            stratify=s1_cluster_labels,
            random_state=config["seed"],
        )
        s1_train_indices = [idx for ci in s1_train_ci for idx in clusters[s1_cluster_ids[ci]]]
        s1_val_indices   = [idx for ci in s1_val_ci   for idx in clusters[s1_cluster_ids[ci]]]

        train_dataset_s1 = make_dataset(all_sequences, all_labels, all_ids, s1_train_indices)
        val_dataset_s1   = make_dataset(all_sequences, all_labels, all_ids, s1_val_indices)
        logger.info(f"Stage 1 train: {len(train_dataset_s1)} seqs | "
                    f"val: {len(val_dataset_s1)} seqs")

        train_loader_s1 = DataLoader(
            train_dataset_s1,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
        )
        val_loader_s1 = DataLoader(
            val_dataset_s1,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
        )

        model = build_contrastive_model(config, device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

        loss_fn_s1 = SupConLoss(temperature=config["training"]["temperature"])
        optimizer_s1 = create_optimizer(model,
                                        learning_rate=config["training"]["learning_rate"],
                                        weight_decay=config["training"]["weight_decay"])
        num_steps = len(train_loader_s1) * config["training"]["num_epochs"]
        scheduler_s1 = create_scheduler(optimizer_s1,
                                        num_training_steps=num_steps,
                                        num_warmup_steps=config["training"]["warmup_steps"],
                                        scheduler_type=config["training"]["scheduler"],
                                        min_lr=config["training"]["min_lr"])
        scaler = (torch.cuda.amp.GradScaler()
                  if config["training"]["mixed_precision"] and device == "cuda" else None)

        early_stopping_s1 = EarlyStopping(
            patience=config["training"]["patience"],
            min_delta=config["training"]["min_delta"],
            mode="min",
        )
        metrics_tracker = MetricsTracker()
        best_val_loss_s1 = float("inf")

        logger.info("Starting Stage 1 (early stopping on val contrastive loss)...")
        for epoch in range(1, config["training"]["num_epochs"] + 1):
            train_loss, train_m = train_epoch(
                model=model, dataloader=train_loader_s1, loss_fn=loss_fn_s1,
                optimizer=optimizer_s1, device=device,
                grad_clip=config["training"]["gradient_clip_norm"], scaler=scaler,
            )
            val_loss, val_m = validate(
                model=model, dataloader=val_loader_s1, loss_fn=loss_fn_s1, device=device,
            )
            if scheduler_s1:
                scheduler_s1.step()

            logger.info(f"[S1] Epoch {epoch:>3}/{config['training']['num_epochs']} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Silhouette: {val_m['silhouette']:.4f} | "
                        f"Davies-Bouldin: {val_m['davies_bouldin']:.4f}")

            metrics_tracker.update({"train_loss": train_loss, "val_loss": val_loss,
                                    "train_silhouette": train_m["silhouette"],
                                    "val_silhouette": val_m["silhouette"],
                                    "train_davies_bouldin": train_m["davies_bouldin"],
                                    "val_davies_bouldin": val_m["davies_bouldin"]})

            # Save periodic checkpoint
            if epoch % config["training"]["save_every_n_epochs"] == 0:
                save_checkpoint(model=model, optimizer=optimizer_s1, scheduler=scheduler_s1,
                                epoch=epoch, metrics=val_m,
                                save_path=str(checkpoint_dir / f"stage1_epoch_{epoch}.pt"))

            # Save best val-loss checkpoint
            if val_loss < best_val_loss_s1:
                best_val_loss_s1 = val_loss
                save_checkpoint(model=model, optimizer=optimizer_s1, scheduler=scheduler_s1,
                                epoch=epoch, metrics=val_m,
                                save_path=str(stage1_ckpt))
                logger.info(f"[S1]   -> Best val loss: {val_loss:.4f} (epoch {epoch})")

            if early_stopping_s1(val_loss):
                logger.info(f"[S1] Early stopping at epoch {epoch}")
                break

        metrics_tracker.save(str(output_dir / "stage1_metrics.json"))
        logger.info(f"[S1] Done. Best val loss: {best_val_loss_s1:.4f} → {stage1_ckpt}")

        # Reload the best (lowest train loss) checkpoint
        ckpt = torch.load(stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Freeze ALL Stage 1 weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Stage 1 weights frozen.")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — 5-fold CV MLP classifier
    # ═════════════════════════════════════════════════════════════════════════
    n_folds = args.n_folds
    logger.info("\n" + "=" * 65)
    logger.info(f"  STAGE 2 — {n_folds}-fold CV MLP Classifier (frozen embeddings)")
    logger.info("=" * 65)

    fold_splits = make_cv_splits(clusters, all_labels, n_folds=n_folds, seed=config["seed"])

    for fs in fold_splits:
        tl = [all_labels[i] for i in fs["train"]]
        vl = [all_labels[i] for i in fs["val"]]
        tsl = [all_labels[i] for i in fs["test"]]
        logger.info(f"  Fold {fs['fold']}: "
                    f"train={len(fs['train'])} (pos={sum(tl)}, neg={len(tl)-sum(tl)}), "
                    f"val={len(fs['val'])} (pos={sum(vl)}, neg={len(vl)-sum(vl)}), "
                    f"test={len(fs['test'])} (pos={sum(tsl)}, neg={len(tsl)-sum(tsl)})")

    clf_input_dim = config["model"]["projector"]["output_dim"]
    clf_hidden_dim = config["training"].get("classifier_hidden_dim", 64)
    clf_dropout = config["training"].get("classifier_dropout", 0.3)
    clf_lr = config["training"].get("classifier_lr", 1e-3)
    clf_epochs = config["training"].get("classifier_epochs", 50)
    clf_patience = config["training"].get("classifier_patience", 10)

    all_fold_results = []

    for fs in fold_splits:
        fold = fs["fold"]
        logger.info("")
        logger.info("=" * 65)
        logger.info(f"  FOLD {fold}/{n_folds}")
        logger.info("=" * 65)

        set_seed(config["seed"] + fold)

        train_dataset = make_dataset(all_sequences, all_labels, all_ids, fs["train"])
        val_dataset = make_dataset(all_sequences, all_labels, all_ids, fs["val"])
        test_dataset = make_dataset(all_sequences, all_labels, all_ids, fs["test"])

        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"],
                                shuffle=False, collate_fn=collate_fn)

        # Class weights from this fold's training labels
        # class_weights = compute_class_weights(train_dataset.labels, device)
        # logger.info(f"[Fold {fold}] Class weights — "
        #             f"Cytosolic: {class_weights[0]:.4f}, Nitroplast: {class_weights[1]:.4f}")
        # loss_fn_s2 = nn.CrossEntropyLoss(weight=class_weights)
        loss_fn_s2 = nn.CrossEntropyLoss()

        # Fresh MLP for each fold
        classifier = MLPClassifier(input_dim=clf_input_dim,
                                   hidden_dim=clf_hidden_dim,
                                   dropout=clf_dropout).to(device)
        optimizer_s2 = torch.optim.Adam(classifier.parameters(), lr=clf_lr, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=clf_patience, min_delta=0.005, mode="max")

        best_val_mcc = -1.0
        best_clf_state = None

        logger.info(f"[Fold {fold}] Training MLP classifier ({clf_epochs} epochs max)...")
        for epoch in range(1, clf_epochs + 1):
            train_loss = train_classifier_epoch(classifier, model, train_loader,
                                                optimizer_s2, loss_fn_s2, device)
            _, val_mcc, val_f1, _, _ = eval_classifier(classifier, model, val_loader,
                                                          loss_fn_s2, device)

            if epoch % 5 == 0 or epoch == 1 or epoch == clf_epochs:
                logger.info(f"[Fold {fold}] Clf epoch {epoch:>3}/{clf_epochs} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val MCC: {val_mcc:.4f} | Val F1: {val_f1:.4f}")

            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_clf_state = {k: v.clone() for k, v in classifier.state_dict().items()}
                logger.info(f"[Fold {fold}]   -> Best classifier (val_mcc={val_mcc:.4f})")

            if early_stopping(val_mcc):
                logger.info(f"[Fold {fold}] Early stopping at epoch {epoch}")
                break

        if best_clf_state:
            classifier.load_state_dict(best_clf_state)

        # Evaluate on test fold
        logger.info(f"[Fold {fold}] Evaluating on test set...")
        y_true_test, y_pred_test = predict_dataset(classifier, model, test_dataset, config, device)
        test_metrics = log_metrics(y_true_test, y_pred_test, logger, tag=f"Fold{fold}/test")

        # Extract and evaluate sub-test
        sub_test_dataset = filter_sub_test(test_dataset, sub_test_id_set)
        if sub_test_dataset:
            logger.info(f"[Fold {fold}] Sub-test: {len(sub_test_dataset)} proteins "
                        f"(pos={sum(sub_test_dataset.labels)}, "
                        f"neg={len(sub_test_dataset)-sum(sub_test_dataset.labels)})")
        else:
            logger.info(f"[Fold {fold}] Sub-test: 0 proteins matched")

        y_true_sub, y_pred_sub = predict_dataset(classifier, model, sub_test_dataset, config, device)
        sub_test_metrics = log_metrics(y_true_sub, y_pred_sub, logger, tag=f"Fold{fold}/sub_test")

        fold_result = {
            "fold": fold,
            "best_val_mcc": float(best_val_mcc),
            "test_metrics": test_metrics,
            "sub_test_metrics": sub_test_metrics,
        }
        all_fold_results.append(fold_result)
        with open(cv_dir / f"fold_{fold}_results.json", "w") as f:
            json.dump(fold_result, f, indent=2)
        logger.info(f"[Fold {fold}] Results saved.")

    # ── CV Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("  STAGE 2 CROSS-VALIDATION SUMMARY")
    logger.info("=" * 65)

    def agg(results, key, sub_key):
        vals = [r[key][sub_key] for r in results if r[key] is not None and sub_key in r[key]]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    summary = {"n_folds": n_folds, "folds": all_fold_results, "aggregate": {}}
    for subset_key, label in [("test_metrics", "Test"), ("sub_test_metrics", "Sub-Test")]:
        logger.info(f"\n  {label} Set:")
        subset_agg = {}
        for metric in ["accuracy", "recall", "f1", "mcc"]:
            mean, std = agg(all_fold_results, subset_key, metric)
            if mean is not None:
                logger.info(f"    {metric:<12}: {mean:.4f} ± {std:.4f}")
                subset_agg[metric] = {"mean": mean, "std": std}
            else:
                logger.info(f"    {metric:<12}: N/A")
        summary["aggregate"][label.lower().replace("-", "_")] = subset_agg

    summary_path = cv_dir / "stage2_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("Training complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-stage training: SupCon pretraining + CV MLP classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recompute of sequence clustering")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Stage 1 and load existing stage1_best.pt")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds for Stage 2 (default: 5)")
    args = parser.parse_args()
    main(args)
