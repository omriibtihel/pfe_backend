"""
NormalityAnalyzer — service central de test de normalité.

Toute la logique statistique vit ici. Les routes et le profiler délèguent
à ce module. Ne pas dupliquer de logique scipy dans d'autres fichiers.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses de résultat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnNormalityResult:
    col: str
    n: int
    mean: float | None
    std: float | None
    skewness: float | None
    excess_kurtosis: float | None          # Convention Fisher — 0 pour une gaussienne
    test_used: str
    stat: float | None
    p_value: float | None
    p_value_corrected: float | None        # BH-ajusté ; None avant correction multi-col
    is_normal_statistical: bool            # p_value_corrected > 0.05
    is_normal_practical: bool              # |skewness| < 0.5 ET |excess_kurtosis| < 2
    distribution_shape: str               # "symmetric" | "approximately_symmetric" |
                                           # "right_skewed" | "left_skewed" | "heavy_tailed"
    skewness_level: str                   # "mild" | "moderate" | "severe"
    has_non_positive: bool                 # data.min() <= 0 → Box-Cox inapplicable
    recommended_transform: str            # "none"|"log"|"sqrt"|"yeo_johnson"|"box_cox"
    error: str | None = None


@dataclass
class NormalityReport:
    results: list[ColumnNormalityResult]
    correction_applied: bool
    k_tested: int
    columns_skipped: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class NormalityAnalyzer:

    ALPHA = 0.05

    # ── Preprocessing ────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_series(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        return s.dropna()

    # ── Sélection du test ────────────────────────────────────────────────────

    @staticmethod
    def _select_test(n: int) -> str:
        if n < 8:
            raise ValueError("insufficient_data")
        if n <= 50:
            return "shapiro"
        if n <= 2000:
            return "dagostino"
        return "anderson"

    # ── Exécution ────────────────────────────────────────────────────────────

    @classmethod
    def _run_test(cls, data: np.ndarray, test: str) -> tuple[float, float]:
        if test == "shapiro":
            stat, p = stats.shapiro(data)
            return float(stat), float(p)
        if test == "dagostino":
            stat, p = stats.normaltest(data)
            return float(stat), float(p)
        # anderson — method='interpolate' retourne res.pvalue directement (SciPy ≥ 1.17)
        res = stats.anderson(data, dist="norm", method="interpolate")
        return float(res.statistic), float(res.pvalue)

    # ── Classification ───────────────────────────────────────────────────────

    @staticmethod
    def _classify_distribution(
        skewness: float, excess_kurtosis: float
    ) -> tuple[str, str]:
        """Retourne (distribution_shape, skewness_level)."""
        abs_sk = abs(skewness)

        if abs_sk < 0.2 and abs(excess_kurtosis) < 1:
            shape = "symmetric"
        elif skewness > 0.5:
            shape = "right_skewed"
        elif skewness < -0.5:
            shape = "left_skewed"
        elif abs(excess_kurtosis) > 3:
            shape = "heavy_tailed"
        else:
            shape = "approximately_symmetric"

        if abs_sk < 0.5:
            level = "mild"
        elif abs_sk < 1.0:
            level = "moderate"
        else:
            level = "severe"

        return shape, level

    # ── Recommandation de transformation ─────────────────────────────────────

    @staticmethod
    def _recommend_transform(
        skewness: float,
        excess_kurtosis: float,
        has_non_positive: bool,
    ) -> str:
        abs_sk = abs(skewness)

        if abs_sk < 0.2 and abs(excess_kurtosis) < 1:
            return "none"

        if skewness > 0.5:                          # asymétrie droite
            if skewness >= 1.0 and not has_non_positive:
                return "log"
            if not has_non_positive:
                return "sqrt"
            return "yeo_johnson"

        if skewness < -0.5:                         # asymétrie gauche
            return "yeo_johnson"

        if abs(excess_kurtosis) > 3:                # distribution lourde
            return "box_cox" if not has_non_positive else "yeo_johnson"

        return "none"

    # ── Analyse colonne unique ────────────────────────────────────────────────

    def analyze_column(self, series: pd.Series, col_name: str) -> ColumnNormalityResult:
        try:
            cleaned = self._prepare_series(series)
            n = len(cleaned)
            test = self._select_test(n)          # lève ValueError si n < 8
            data = cleaned.to_numpy(dtype=float)

            mean = float(np.mean(data))
            std = float(np.std(data, ddof=1))
            skewness = float(stats.skew(data))
            excess_kurtosis = float(stats.kurtosis(data))  # Fisher : 0 pour N(0,1)
            has_non_positive = bool(float(data.min()) <= 0)

            stat, p_value = self._run_test(data, test)
            shape, level = self._classify_distribution(skewness, excess_kurtosis)
            transform = self._recommend_transform(skewness, excess_kurtosis, has_non_positive)

            is_normal_practical = abs(skewness) < 0.5 and abs(excess_kurtosis) < 2.0

            return ColumnNormalityResult(
                col=col_name,
                n=n,
                mean=mean,
                std=std,
                skewness=skewness,
                excess_kurtosis=excess_kurtosis,
                test_used=test,
                stat=stat,
                p_value=p_value,
                p_value_corrected=None,
                is_normal_statistical=bool(p_value > self.ALPHA),
                is_normal_practical=is_normal_practical,
                distribution_shape=shape,
                skewness_level=level,
                has_non_positive=has_non_positive,
                recommended_transform=transform,
                error=None,
            )

        except ValueError as exc:
            return ColumnNormalityResult(
                col=col_name,
                n=0,
                mean=None,
                std=None,
                skewness=None,
                excess_kurtosis=None,
                test_used="",
                stat=None,
                p_value=None,
                p_value_corrected=None,
                is_normal_statistical=False,
                is_normal_practical=False,
                distribution_shape="unknown",
                skewness_level="",
                has_non_positive=False,
                recommended_transform="none",
                error=str(exc),
            )

    # ── Analyse multi-colonnes avec correction BH ─────────────────────────────

    def analyze_columns(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> NormalityReport:
        results: list[ColumnNormalityResult] = []
        skipped: list[str] = []

        for col in columns:
            res = self.analyze_column(df[col], col)
            if res.error:
                skipped.append(col)
            results.append(res)

        valid = [r for r in results if not r.error]
        k = len(valid)
        correction_applied = k > 1

        if correction_applied:
            sorted_valid = sorted(valid, key=lambda r: r.p_value)  # type: ignore[arg-type]
            p_arr = np.array([r.p_value for r in sorted_valid], dtype=float)
            adjusted = _bh_adjust(p_arr)
            for r, p_adj in zip(sorted_valid, adjusted):
                r.p_value_corrected = float(p_adj)
                r.is_normal_statistical = bool(float(p_adj) > self.ALPHA)
        else:
            for r in valid:
                r.p_value_corrected = r.p_value

        return NormalityReport(
            results=results,
            correction_applied=correction_applied,
            k_tested=k,
            columns_skipped=skipped,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Correction Benjamini-Hochberg (step-up). Entrée triée ascendante."""
    k = len(p_values)
    if k == 0:
        return p_values
    adjusted = np.empty(k)
    adjusted[-1] = p_values[-1]
    for i in range(k - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], p_values[i] * k / (i + 1))
    return np.minimum(adjusted, 1.0)


def report_to_dict(report: NormalityReport) -> dict:
    """Convertit NormalityReport en dict JSON-sérialisable (nan → None)."""
    def _clean(val):
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val

    serialized_results = []
    for r in report.results:
        d = dataclasses.asdict(r)
        serialized_results.append({k: _clean(v) for k, v in d.items()})

    return {
        "results": serialized_results,
        "correction_applied": report.correction_applied,
        "k_tested": report.k_tested,
        "columns_skipped": report.columns_skipped,
    }
