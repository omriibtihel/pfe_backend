"""
LIME explanation services.

Public API
----------
compute_local_lime(fitted_pipe, row, *, task_type, background, feature_names)
    → list | None   (returned in prediction response as row["lime"])

Mirrors the contract of app.services.shap.compute_local_shap so the predictor
layer can call either method through a uniform interface.
"""
from app.services.lime.local_lime import compute_local_lime

__all__ = ["compute_local_lime"]
