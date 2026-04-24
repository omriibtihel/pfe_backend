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

---

## ~~CHAIN 3 — Parameter integrity~~ ✅ FIXED

**Root problem:** `int(payload.get("x", DEFAULT) or DEFAULT)` silently replaced any falsy
value (0, False, "") with the default. `random_state=0` became 42, `testRatio=0` became 20.
No log, no warning, artifacts did not show the value actually used.

**Fixed in:**
- `app/services/training/config/schema/training_config.py` — all 5 int params now use
  `_int_or_default(raw, default)` (None-check, never `or DEFAULT`): kFolds, nIterRandomSearch,
  gridCvFolds/innerCvFolds, **randomState** (CRITICAL — 0 is a valid seed), nRepeats
- `app/services/training/config/schema/preprocessing.py` — `varianceThreshold=0` (disable
  variance filtering) was silently replaced by 0.01; fixed to None-check
- `resolved_params` dict added to `TrainingConfig` and included in `as_dict()` so every
  session artifact records the values actually used
- `TrainingConfig.validate()` added — raises `ValueError` for: kFolds < 2 (CV), testRatio > 95,
  ratio sum ≠ 100 (holdout), nRepeats < 1 (repeated CV)
- `app/api/routes/preparation.py` — `validate_training`: `ValueError` → HTTP 422
- `app/api/routes/training.py` — `start_training_for_version`: eager `from_front()` call
  before background task so param errors return 422 not silent failure

**Tests (30 new):**
- `tests/services/training/test_training_config_params.py` — 11 tests
- `tests/services/training/test_training_config_validation.py` — 11 tests
- `tests/services/training/test_training_config_bool_params.py` — 8 tests

---

## CHAIN 4 — Frontend silent errors (OPEN — out of scope for Chain 3 session)

These frontend locations silently swallow errors. No user feedback, no console warning.
Must be fixed before the next release.

### FE-01: PredictionResultsPage.tsx:296 — empty catch on JSON.parse

```typescript
try {
  data = JSON.parse(raw);
} catch {
  // silent — user sees stale or empty data
}
```

Fix: log the parse error and show an error toast.

---

### FE-02: useNettoyageData.ts:235-243 — `.catch(() => null)` on column metadata

Column metadata fetch silently returns `null` on failure. The cleaning UI loads with
no column info and no indication that something failed.

Fix: surface the error to the user instead of swallowing it.

---

### FE-03: TrainingPage.tsx:247 — silent history load failure

Training history load failure is caught and discarded. The page renders as if there
is no history rather than indicating a fetch problem.

Fix: show an error state or retry indicator.

---

### FE-04: trainingService.ts:507 — unhandled rejection on download

The download promise rejection is not caught. This can produce an unhandled promise
rejection in the browser console with no user-visible feedback.

Fix: add `.catch()` with a toast or error state.
