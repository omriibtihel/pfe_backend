# FOLLOWUP_BUGS.md

Bugs identified during the Cleaning Illusion fix (Chain 1) that are **out of scope** for this session.
These must be fixed before the next release.

---

## ~~BUG-01: AutoML session also loads raw CSV~~ ✅ FIXED

**Fixed in:** `app/services/training/training_service.py` — `run_automl_session()` now
calls `resolve_training_data_path()` instead of `load_dataframe(dataset_path)` directly.

**Shared helper:** `app/services/training/data_source.py::resolve_training_data_path()`
encapsulates the processed-file preference logic for all training-adjacent data loads.

---

## ~~BUG-02: Validation preview also loads raw CSV~~ ✅ FIXED

**Fixed in:** `app/api/routes/preparation.py`
- `validate_training` (line 158): now calls `resolve_training_data_path()`
- `analyze_balance` (line 200): now calls `resolve_training_data_path()`

Both routes now use the same data source as the training background task.

---

## ~~BUG-03: Training validation route also loads raw CSV~~ ✅ FIXED

**Fixed in:** `app/api/routes/training.py` — `start_training_for_version()` pre-flight
balance check (line 171) now calls `resolve_training_data_path()` so it evaluates
class distribution on the same data that training will use.

---

## ~~BUG-04: presenter.py silently returns value=0.0 on metric parse failure~~ ✅ FIXED

**Fixed in:** `app/services/training/presenter.py` + `app/schemas/training/results.py`

- `PrimaryMetric` now has `value: Optional[float]` and `status: Literal["success","not_applicable","error"]`
- `get_primary_metric()` wrapped in outer try/except; raises `MetricNotApplicable` when no
  metric found → `status="not_applicable"`; unexpected exceptions → `status="error"` + log
- `ModelResultCard.tsx`: red badge on error, muted "N/A" on not_applicable, unchanged on success

---

## ~~BUG-05: testScore semantically overloaded between holdout and CV-only mode~~ ✅ FIXED

**Fixed in:** `app/services/training/orchestrator.py` + `presenter.py` + `ModelResultCard.tsx`

- orchestrator adds `test_is_cv_mean: bool` and `test_label: str` to the CV metrics_json
- presenter forwards both as `testIsCvMean` / `testLabel` on `ModelResultResponse`
- TypeScript `ModelResult` interface updated with optional `testIsCvMean` and `testLabel`
- `ModelResultCard` shows "CV val." underline with tooltip when `testIsCvMean` is true
