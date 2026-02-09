"""
ESM-2 encoder with LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

This module wraps the pretrained ESM-2 model and adds trainable low-rank
adapters to the attention layers.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESMEncoder(nn.Module):
    """
    ESM-2 encoder with optional LoRA adaptation.
    
    The encoder:
    1. Tokenizes amino acid sequences
    2. Passes through ESM-2 transformer
    3. Pools residue embeddings to get protein-level representation
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        freeze_layers: Optional[int] = None,
        pooling_method: str = "mean",
        device: str = "cuda"
    ):
        """
        Args:
            model_name: HuggingFace model identifier for ESM-2
            use_lora: Whether to add LoRA adapters
            lora_config: LoRA hyperparameters (r, alpha, dropout, target_modules)
            freeze_layers: Number of transformer layers to freeze (from bottom)
            pooling_method: How to pool residue embeddings (mean, max, attention_weighted)
            device: cuda or cpu
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.device = device
        
        # Load tokenizer and model
        logger.info(f"Loading ESM-2 model: {model_name}")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm_model = EsmModel.from_pretrained(model_name)
        
        # Get embedding dimension
        self.embedding_dim = self.esm_model.config.hidden_size
        logger.info(f"ESM-2 embedding dimension: {self.embedding_dim}")
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config or {})
        
        # Freeze layers if requested
        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)
        
        # Attention-weighted pooling (if needed)
        if pooling_method == "attention_weighted":
            self.pooling_attention = nn.Linear(self.embedding_dim, 1)
        
        self.to(device)
    
    def _apply_lora(self, lora_config: Dict):
        """Add LoRA adapters to the model."""
        logger.info("Applying LoRA adaptation...")
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ["query", "value"]),
            bias="none"
        )
        
        self.esm_model = get_peft_model(self.esm_model, peft_config)
        self.esm_model.print_trainable_parameters()
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first N transformer layers."""
        logger.info(f"Freezing first {num_layers} layers...")
        
        # Freeze embeddings
        for param in self.esm_model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified encoder layers
        for i, layer in enumerate(self.esm_model.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def pool_residue_embeddings(
        self,
        residue_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool per-residue embeddings to get protein-level representation.
        
        Args:
            residue_embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Expand mask for broadcasting
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        
        if self.pooling_method == "mean":
            # Average over non-padding positions
            masked_embeddings = residue_embeddings * mask
            pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        elif self.pooling_method == "max":
            # Max pooling (ignoring padding)
            masked_embeddings = residue_embeddings.masked_fill(mask == 0, float('-inf'))
            pooled = masked_embeddings.max(dim=1)[0]
        
        elif self.pooling_method == "attention_weighted":
            # Learnable attention pooling
            attention_weights = self.pooling_attention(residue_embeddings)  # [batch, seq_len, 1]
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
            attention_weights = torch.softmax(attention_weights, dim=1)
            pooled = (residue_embeddings * attention_weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        return pooled
    
    def forward(
        self,
        sequences: List[str],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Encode protein sequences.
        
        Args:
            sequences: List of amino acid sequences
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - embeddings: [batch_size, hidden_dim] - Pooled protein embeddings
                - residue_embeddings: [batch_size, seq_len, hidden_dim] - Per-residue (optional)
                - attention: [batch_size, num_heads, seq_len, seq_len] - Attention weights (optional)
        """
        # Tokenize sequences
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # ESM-2 max length
        ).to(self.device)
        
        # Forward pass through ESM-2
        outputs = self.esm_model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            output_hidden_states=True,
            output_attentions=return_attention
        )
        
        # Get last layer hidden states
        residue_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Pool to protein-level
        protein_embeddings = self.pool_residue_embeddings(
            residue_embeddings,
            encoded['attention_mask']
        )
        
        result = {
            'embeddings': protein_embeddings,
            'residue_embeddings': residue_embeddings,
            'attention_mask': encoded['attention_mask']
        }
        
        if return_attention:
            result['attention'] = outputs.attentions  # Tuple of attention matrices
        
        return result
    
    def get_attention_weights(
        self,
        sequences: List[str],
        layer_idx: int = -1,
        aggregate_heads: str = "mean"
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability.
        
        Args:
            sequences: List of amino acid sequences
            layer_idx: Which transformer layer (-1 for last)
            aggregate_heads: How to combine attention heads (mean, max, or None)
        
        Returns:
            attention: [batch_size, seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len]
        """
        outputs = self.forward(sequences, return_attention=True)
        
        # Get attention from specified layer
        attention = outputs['attention'][layer_idx]  # [batch, num_heads, seq_len, seq_len]
        
        if aggregate_heads == "mean":
            attention = attention.mean(dim=1)  # Average across heads
        elif aggregate_heads == "max":
            attention = attention.max(dim=1)[0]  # Max across heads
        elif aggregate_heads is None:
            pass  # Keep all heads
        else:
            raise ValueError(f"Unknown aggregation: {aggregate_heads}")
        
        return attention
    
    def save_pretrained(self, save_path: str):
        """Save the model."""
        logger.info(f"Saving ESM encoder to {save_path}")
        self.esm_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """Load the model."""
        logger.info(f"Loading ESM encoder from {load_path}")
        self.esm_model = EsmModel.from_pretrained(load_path)
        self.tokenizer = EsmTokenizer.from_pretrained(load_path)
        self.to(self.device)


if __name__ == "__main__":
    # Test the encoder
    encoder = ESMEncoder(
        model_name="facebook/esm2_t33_650M_UR50D",
        use_lora=True,
        lora_config={'r': 8, 'alpha': 16, 'dropout': 0.1},
        freeze_layers=20,
        pooling_method="mean"
    )
    
    # Test sequences
    test_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    ]
    
    # Forward pass
    outputs = encoder(test_sequences)
    print(f"Protein embedding shape: {outputs['embeddings'].shape}")
    print(f"Residue embeddings shape: {outputs['residue_embeddings'].shape}")
    
    # Test attention extraction
    attention = encoder.get_attention_weights(test_sequences)
    print(f"Attention weights shape: {attention.shape}")
