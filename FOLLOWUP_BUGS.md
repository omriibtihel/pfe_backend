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

## BUG-04: presenter.py silently returns value=0.0 on metric parse failure

**File:** `app/services/training/presenter.py:85-172`
**Audit reference:** Deception Audit AXE 3

Four nested try/except blocks in `get_primary_metric()` swallow any parsing
failure and return `PrimaryMetric(name="unknown", value=0.0, displayName="—")`.
The user sees "—" in the ModelResultCard and cannot distinguish "metric not
applicable for this model" from "metric calculation failed entirely."

**Fix:** Return `PrimaryMetric(name="error", value=None, displayName="Erreur calcul")`
on parse failure and display a red badge in the frontend.

---

## BUG-05: testScore semantically overloaded between holdout and CV-only mode

**File:** `app/services/training/orchestrator.py:1557` + `presenter.py:207-248`
**Audit reference:** Deception Audit AXE 3

When `test_ratio=0` (CV without holdout), `testScore` = mean of CV validation
fold metrics, but is presented identically to a true holdout test score.
Frontend cannot distinguish the two cases.

**Fix:** Add `"test_is_cv_mean": true` flag in the API response and update
`ModelResultCard.tsx` to display "CV val." instead of the primary metric label.
