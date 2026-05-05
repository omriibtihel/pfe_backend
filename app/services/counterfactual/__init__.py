"""
Counterfactual explanation services.

Public API
----------
compute_counterfactual(fitted_pipe, row, *, features_to_vary, task_type,
                       background, feature_names, n_counterfactuals, random_state)
    → list | None   (returned in prediction response as row["counterfactual"])

Mirrors the contract of app.services.shap and app.services.lime so the
predictor layer can call all three methods through a uniform interface.
"""
from app.services.counterfactual.local_counterfactual import compute_counterfactual

__all__ = ["compute_counterfactual"]
