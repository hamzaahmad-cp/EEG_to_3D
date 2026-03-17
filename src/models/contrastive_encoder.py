"""
Contrastive EEG Encoder for CATVis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple

from src.models.eeg_classifier import _make_conformer


class ContrastiveEncoder(nn.Module):
    """
    EEG Encoder for contrastive learning with text embeddings.
    Based on EEGConformer, modified to output 768-dimensional embeddings
    aligned with CLIP text embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        super(ContrastiveEncoder, self).__init__()
        self.config = config

        model_config = config['eeg_classification']
        model_params = model_config['model_params']

        self.model = _make_conformer(model_config, model_params)

        # Modify final layers to produce 768-dimensional features (CLIP-aligned)
        self.model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
        self.model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=model_config['n_outputs'], bias=True)

        # Remove final classification layer, leaving us with 768-d embeddings
        self.model.final_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            return out[1]  # embeddings
        return out

    def load_pretrained_weights(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, checkpoint_path: str):
        torch.save(self.model.state_dict(), checkpoint_path)


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
