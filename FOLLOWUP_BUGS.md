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

## ~~CHAIN 4 — Frontend silent errors~~ ✅ FIXED

All four silent-error locations now have visible, distinguishable error states.
Core principle enforced: loading / error / legitimately-empty are visually distinct.

### ~~FE-01: PredictionResultsPage.tsx — parse error now surfaced~~ ✅ FIXED

**Fixed in:** `src/pages/project/PredictionResultsPage.tsx`
- `parseError: boolean` state added, reset to false before each sessionStorage read
- `catch (e)` logs + `toast({ variant: "destructive" })` + `setParseError(true)`
- New `data-testid="prediction-parse-error"` render state shows distinct error UI
  (AlertTriangle icon, message, "Nouvelle prédiction" button)
- `data-testid="prediction-empty"` marks the legitimate "no prediction yet" state
- Three states now visually distinct: loading (Loader2 spinner) / error (red alert)
  / empty (Target icon + muted message)

---

### ~~FE-02: useNettoyageData.ts — column metadata error surfaced~~ ✅ FIXED

**Fixed in:** `src/pages/project/nettoyage/useNettoyageData.ts` +
             `src/pages/project/NettoyagePage.tsx`
- `columnsError: string | null` state added to hook
- `.catch(() => null)` replaced with flag-based tracking: `metaFetchFailed` captured
  in `refreshProcessing`, `setColumnsError` called after `Promise.all`
- `retryColumnsLoad` callback added — clears error and re-calls `refreshProcessing`
- Hook return shape extended with `columnsError` and `retryColumnsLoad`
- `NettoyagePage` shows persistent inline error banner (`data-testid="columns-error-banner"`)
  with "Réessayer" button (`data-testid="columns-error-retry"`) — NOT a toast (blocking failure)

---

### ~~FE-03: TrainingPage.tsx — history failure shows soft non-blocking indicator~~ ✅ FIXED

**Fixed in:** `src/pages/project/TrainingPage.tsx`
- `historyLoadFailed: boolean` state added
- `.catch(() => {})` replaced with `console.warn` + `setHistoryLoadFailed(true)`
- Small muted banner rendered when `historyLoadFailed` (`data-testid="history-load-failed"`):
  History icon + "Historique indisponible" + compact "Réessayer" button
  (`data-testid="history-retry-btn"`)
- Training launch wizard is NOT blocked — wizard stepper and buttons remain active

---

### ~~FE-04: trainingService.ts — downloadResults never rejects~~ ✅ FIXED

**Fixed in:** `src/services/trainingService.ts`
- `downloadResults()` return type changed from `Promise<Blob>` to
  `Promise<{ success: true; blob: Blob } | { success: false; error: string }>`
- Both the primary `/download` and fallback `/export` paths are wrapped in
  `try/catch` — if both fail, resolves `{ success: false, error: String(e) }`,
  never throws
- `downloadResultsAndSaveToDisk()` checks `result.success` and throws
  `new Error(result.error)` on failure so `TrainingResultsPage`'s existing
  `catch + toast` still fires

---

## DECEPTION AUDIT — Formal Closure

| Chain | Theme | Count | Severity | Status |
|---|---|---|---|---|
| Chain 1 | Cleaning illusion (raw CSV loaded after cleaning) | 3 | CRITICAL | ✅ FIXED |
| Chain 1+ | Clone call sites (automl, validation preview, balance check) | 3 | WARNING | ✅ FIXED |
| Chain 2 | Metrics integrity (presenter, testScore overload) | 2 | CRITICAL | ✅ FIXED |
| Chain 3 | Parameter integrity (falsy-default, bool audit, validate()) | 3 | CRITICAL | ✅ FIXED |
| Chain 4 | Frontend silences (4 locations, 13 tests) | 4 | WARNING | ✅ FIXED |

**Total issues resolved:** 15 across 5 chains, 58 new regression tests added.
