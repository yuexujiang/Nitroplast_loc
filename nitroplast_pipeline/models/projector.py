"""
Projection head for supervised contrastive learning.

Maps ESM-2 embeddings to a lower-dimensional space optimized for
distinguishing nitroplast-localized from cytosolic proteins.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Architecture: Linear -> BatchNorm -> ReLU -> Dropout -> ... -> Linear
    """
    
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Dimension of ESM-2 embeddings
            hidden_dims: List of hidden layer dimensions
            output_dim: Final embedding dimension
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Batch normalization (except last layer)
            if use_batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Activation (except last layer)
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        
        self.projector = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to contrastive space.
        
        Args:
            embeddings: [batch_size, input_dim]
        
        Returns:
            projected: [batch_size, output_dim]
        """
        projected = self.projector(embeddings)
        
        # L2 normalize for contrastive learning
        projected = nn.functional.normalize(projected, p=2, dim=1)
        
        return projected


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss (Khosla et al., NeurIPS 2020).
    
    This loss:
    1. Pulls together embeddings of the same class
    2. Pushes apart embeddings of different classes
    
    Formula:
    L = -log [ Σ_p exp(z·z_p / τ) / Σ_a exp(z·z_a / τ) ]
    
    where:
    - z is the anchor embedding
    - z_p are positive examples (same class)
    - z_a are all other examples in the batch
    - τ is the temperature parameter
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07
    ):
        """
        Args:
            temperature: Softmax temperature for contrastive learning
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: [batch_size, embedding_dim] - L2 normalized embeddings
            labels: [batch_size] - Class labels (0 or 1)
            mask: Optional [batch_size, batch_size] - Manual similarity mask
        
        Returns:
            loss: Scalar contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [batch_size, embedding_dim]')
        
        # Ensure labels are the right shape
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # Create mask: 1 if same class, 0 otherwise
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # Compute similarity matrix: [batch_size, batch_size]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude self-contrast (diagonal)
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class NitroplastContrastiveModel(nn.Module):
    """
    Complete model: ESM-2 Encoder + Projection Head.
    
    This wraps the encoder and projector for end-to-end training.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projector: ProjectionHead
    ):
        """
        Args:
            encoder: ESMEncoder instance
            projector: ProjectionHead instance
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
    
    def forward(
        self,
        sequences: List[str],
        return_attention: bool = False
    ) -> dict:
        """
        Full forward pass.
        
        Args:
            sequences: List of amino acid sequences
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with:
                - embeddings: ESM-2 pooled embeddings
                - projected: Contrastive embeddings
                - residue_embeddings: Per-residue embeddings (optional)
                - attention: Attention weights (optional)
        """
        # Encode sequences
        encoder_outputs = self.encoder(sequences, return_attention=return_attention)
        
        # Project to contrastive space
        projected = self.projector(encoder_outputs['embeddings'])
        
        outputs = {
            'embeddings': encoder_outputs['embeddings'],  # Original ESM-2 embeddings
            'projected': projected,  # Contrastive embeddings (for loss computation)
            'residue_embeddings': encoder_outputs['residue_embeddings']
        }
        
        if return_attention:
            outputs['attention'] = encoder_outputs['attention']
        
        return outputs
    
    def get_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Get final contrastive embeddings for inference.
        
        Args:
            sequences: List of amino acid sequences
        
        Returns:
            embeddings: [batch_size, output_dim] - L2 normalized
        """
        outputs = self.forward(sequences, return_attention=False)
        return outputs['projected']
    
    def save(self, save_path: str):
        """Save the complete model."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'encoder_config': {
                'model_name': self.encoder.model_name,
                'pooling_method': self.encoder.pooling_method,
                'embedding_dim': self.encoder.embedding_dim
            },
            'projector_config': {
                'input_dim': self.projector.input_dim,
                'output_dim': self.projector.output_dim
            }
        }, save_path)
    
    def load(self, load_path: str):
        """Load the complete model."""
        checkpoint = torch.load(load_path, map_location=self.encoder.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])


if __name__ == "__main__":
    # Test projection head
    projector = ProjectionHead(
        input_dim=1280,
        hidden_dims=[512, 256],
        output_dim=128,
        dropout=0.1,
        use_batch_norm=True
    )
    
    # Test input
    test_embeddings = torch.randn(8, 1280)  # Batch of 8
    projected = projector(test_embeddings)
    print(f"Projected embeddings shape: {projected.shape}")
    print(f"L2 norm (should be ~1.0): {projected.norm(dim=1).mean().item():.4f}")
    
    # Test SupCon loss
    loss_fn = SupConLoss(temperature=0.07)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])  # 4 positive, 4 negative
    
    loss = loss_fn(projected, labels)
    print(f"SupCon loss: {loss.item():.4f}")
