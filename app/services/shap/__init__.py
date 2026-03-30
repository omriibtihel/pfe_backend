"""
SHAP explanation services.

Public API
----------
compute_global_shap(fitted_pipe, X_test_raw, *, task_type, feature_names)
    → dict | None   (stored in artifacts["shap"])

compute_local_shap(fitted_pipe, row, *, task_type, background, feature_names)
    → list | None   (returned in prediction response)
"""
from app.services.shap.global_shap import compute_global_shap
from app.services.shap.local_shap import compute_local_shap

__all__ = ["compute_global_shap", "compute_local_shap"]
