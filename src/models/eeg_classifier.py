"""
EEG Classification model for CATVis.
Uses braindecode EEGConformer directly without modifying internal layers.
"""

import torch
import torch.nn as nn
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple


class EEGClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config['eeg_classification']
        model_params = model_config['model_params']

        # Build EEGConformer — outputs n_outputs directly
        self.conformer = EEGConformer(
            n_outputs=model_config['n_outputs'],
            n_chans=model_config['n_chans'],
            n_times=model_config['n_times'],
            n_filters_time=model_params['n_filters_time'],
            filter_time_length=model_params['filter_time_length'],
            pool_time_length=model_params['pool_time_length'],
            pool_time_stride=model_params['pool_time_stride'],
            final_fc_length=model_params['final_fc_length'],
        )

        # Find the embedding dim from the conformer's FC block
        # and add a projection to 768 for CLIP alignment
        self.embed_proj = nn.Linear(model_config['n_outputs'], 768)
        self.classifier_head = nn.Linear(768, model_config['n_outputs'])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Conformer outputs [B, n_outputs]
        logits = self.conformer(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        # Project to 768-dim CLIP space
        embeddings = self.embed_proj(logits)
        # Classify from embeddings
        outputs = self.classifier_head(embeddings)
        return outputs, embeddings

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _, emb = self.forward(x)
        return emb
