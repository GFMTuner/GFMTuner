
import torch
import numpy as np
from typing import Dict, Any


def get_feature_stats(data) -> Dict[str, Any]:
    if not hasattr(data, 'x') or data.x is None:
        return {
            'overall_missing_ratio': 1.0,
            'per_dim_missing_ratio': []
        }
    
    features = data.x.float()
    

    is_nan = torch.isnan(features)
    is_zero = (features.abs() < 1e-8)
    is_missing = is_nan | is_zero
    

    overall_missing = is_missing.sum().item() / features.numel()
    

    per_dim_missing = is_missing.float().mean(dim=0).tolist()
    
    return {
        'overall_missing_ratio': overall_missing,
        'per_dim_missing_ratio': per_dim_missing,
        'num_dims_with_missing': sum(1 for x in per_dim_missing if x > 0.1)
    }
