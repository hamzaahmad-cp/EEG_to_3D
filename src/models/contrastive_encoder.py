"""
Contrastive EEG Encoder for CATVis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple


class ContrastiveEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config['eeg_classification']
        model_params = model_config['model_params']

        # Same as classifier: n_outputs=768 for CLIP-aligned embeddings
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
        # No classification head — just output 768-dim embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.conformer(x)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        return embeddings


def clip_style_contrastive_loss(eeg_embeds: torch.Tensor, text_embeds: torch.Tensor,
                               temperature: float = 0.07) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = eeg_embeds.size(0)
    logits_eeg = eeg_embeds @ text_embeds.t() / temperature
    logits_text = text_embeds @ eeg_embeds.t() / temperature
    labels = torch.arange(batch_size, device=eeg_embeds.device)
    loss_eeg = F.cross_entropy(logits_eeg, labels)
    loss_text = F.cross_entropy(logits_text, labels)
    loss = (loss_eeg + loss_text) / 2.0
    acc_eeg = (logits_eeg.argmax(dim=-1) == labels).float().mean()
    acc_text = (logits_text.argmax(dim=-1) == labels).float().mean()
    metrics = {"loss": loss.item(), "acc_eeg": acc_eeg.item(), "acc_text": acc_text.item()}
    return loss, metrics
