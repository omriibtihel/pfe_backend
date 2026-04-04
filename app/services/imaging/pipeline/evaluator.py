"""
ImagingEvaluator — calcule les métriques finales sur le set de test.

Appelé une fois après la fin de l'entraînement, sur le meilleur checkpoint.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ImagingEvaluator:
    """
    Calcule les métriques de classification sur un DataLoader de test.

    Utilise scikit-learn pour les métriques (déjà présent dans les dépendances).
    """

    def evaluate_classification(
        self,
        model: nn.Module,
        loader: DataLoader,
        class_names: List[str],
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Retourne un dict avec :
          accuracy, f1_macro, f1_weighted, precision_macro, recall_macro,
          roc_auc (OvR), confusion_matrix (list of lists),
          per_class: {class_name: {precision, recall, f1, support}},
          n_samples
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            confusion_matrix,
            classification_report,
        )

        model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[List[float]] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.extend(probs.tolist())

        if not all_labels:
            logger.warning("ImagingEvaluator: aucun échantillon dans le test loader.")
            return {"error": "empty_test_set"}

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        num_classes = len(class_names)

        # ── Métriques de base ─────────────────────────────────────────────────
        acc = float(accuracy_score(y_true, y_pred))
        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_wei = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        prec_mac = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_mac = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        # ── ROC-AUC ───────────────────────────────────────────────────────────
        roc_auc = None
        try:
            if num_classes == 2:
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception as exc:
            logger.warning("ROC-AUC non calculable : %s", exc)

        # ── Matrice de confusion ──────────────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        # ── Rapport par classe ────────────────────────────────────────────────
        report_dict = classification_report(
            y_true, y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        per_class = {
            cls: {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall":    round(float(report_dict[cls]["recall"]), 4),
                "f1":        round(float(report_dict[cls]["f1-score"]), 4),
                "support":   int(report_dict[cls]["support"]),
            }
            for cls in class_names
            if cls in report_dict
        }

        return {
            "accuracy":        round(acc, 4),
            "f1_macro":        round(f1_mac, 4),
            "f1_weighted":     round(f1_wei, 4),
            "precision_macro": round(prec_mac, 4),
            "recall_macro":    round(rec_mac, 4),
            "roc_auc":         round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm.tolist(),
            "per_class":       per_class,
            "n_samples":       int(len(y_true)),
        }
