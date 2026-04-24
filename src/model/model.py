"""
model.py — BiEncoderClassifier architecture.
Encodes resume and JD separately, combines embeddings, classifies into 3 classes.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BiEncoderClassifier(nn.Module):
    """
    Architecture:
        resume_text  → encoder → embedding_r (hidden_size)
        jd_text      → encoder → embedding_j (hidden_size)

        combined = concat(
            embedding_r,
            embedding_j,
            |embedding_r - embedding_j|,
            embedding_r * embedding_j
        )
        = 4 * hidden_size

        classifier_head = Linear(4*hidden_size, 256)
                        → ReLU
                        → Dropout
                        → Linear(256, 3)
    """

    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()

        self.encoder   = AutoModel.from_pretrained(model_name)
        hidden_size    = self.encoder.config.hidden_size
        combined_size  = hidden_size * 4

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool token embeddings using attention mask."""
        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        """Run encoder and mean pool output."""
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.mean_pool(output.last_hidden_state, attention_mask)

    def forward(
        self,
        resume_input_ids:      torch.Tensor,
        resume_attention_mask: torch.Tensor,
        jd_input_ids:          torch.Tensor,
        jd_attention_mask:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits of shape (batch_size, 3).
        """
        emb_r = self.encode(resume_input_ids, resume_attention_mask)
        emb_j = self.encode(jd_input_ids, jd_attention_mask)

        combined = torch.cat([
            emb_r,
            emb_j,
            torch.abs(emb_r - emb_j),
            emb_r * emb_j,
        ], dim=1)

        return self.classifier(combined)