"""
EEG Classification model for CATVis.
"""

import torch
import torch.nn as nn
from braindecode.models import EEGConformer
from typing import Dict, Any, Tuple


def _make_conformer(model_config, model_params):
    """Create EEGConformer, handling API differences across braindecode versions."""
    base = dict(
        n_outputs=model_config['n_outputs'],
        n_chans=model_config['n_chans'],
        n_times=model_config['n_times'],
        n_filters_time=model_params['n_filters_time'],
        filter_time_length=model_params['filter_time_length'],
        pool_time_length=model_params['pool_time_length'],
        pool_time_stride=model_params['pool_time_stride'],
        final_fc_length=model_params['final_fc_length'],
    )
    # Try with all legacy params first, then strip unsupported ones
    optional = {'add_log_softmax': False, 'return_features': True}
    for attempt in range(len(optional) + 1):
        try:
            return EEGConformer(**base, **optional)
        except TypeError as e:
            bad = str(e).split("'")[-2] if "'" in str(e) else None
            if bad and bad in optional:
                optional.pop(bad)
            else:
                raise
    return EEGConformer(**base)


class EEGClassifier(nn.Module):
    """
    EEG Classifier based on EEGConformer from braindecode.
    Modified to produce CLIP-aligned 768-dimensional features.
    """

    def __init__(self, config: Dict[str, Any]):
        super(EEGClassifier, self).__init__()
        self.config = config

        model_config = config['eeg_classification']
        model_params = model_config['model_params']

        self.model = _make_conformer(model_config, model_params)

        # Modify final layers to produce 768-dimensional features (CLIP-aligned)
        self.model.fc.fc[3] = nn.Linear(in_features=256, out_features=768, bias=True)
        self.model.final_layer.final_layer[0] = nn.Linear(in_features=768, out_features=model_config['n_outputs'], bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0], out[1]
        # Newer braindecode may return single tensor — split via hooks
        return out, out

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _, embeddings = self.forward(x)
        return embeddings

    def load_pretrained_weights(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, checkpoint_path: str):
        torch.save(self.model.state_dict(), checkpoint_path)
