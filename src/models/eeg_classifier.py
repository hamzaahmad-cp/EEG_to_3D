"""
EEG Classification model for CATVis.
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

        # Set n_outputs=768 so conformer outputs CLIP-sized embeddings directly
        # This avoids the 40-dim bottleneck that killed accuracy
        self.conformer = EEGConformer(
            n_outputs=768,
            n_chans=model_config['n_chans'],
            n_times=model_config['n_times'],
            n_filters_time=model_params['n_filters_time'],
            filter_time_length=model_params['filter_time_length'],
            pool_time_length=model_params['pool_time_length'],
            pool_time_stride=model_params['pool_time_stride'],
            final_fc_length=model_params['final_fc_length'],
        )

        # Classification head on top of 768-dim embeddings
        self.classifier_head = nn.Linear(768, model_config['n_outputs'])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.conformer(x)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        outputs = self.classifier_head(embeddings)
        return outputs, embeddings

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _, emb = self.forward(x)
        return emb
