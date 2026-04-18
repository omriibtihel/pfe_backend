"""
FeatureEngineeringTransformer
=============================
Sklearn-compatible transformer that creates new columns from user-defined
Python-like expressions before the ColumnTransformer preprocessing step.

Anti-leakage contract
---------------------
- fit()           : validates expressions and computes topological order (stateless)
- transform()     : evaluates expressions using a safe AST evaluator
- fit_transform() : fit then transform (sklearn default)

Expression syntax
-----------------
Write expressions as you would in Python, referencing column names as variables:

    BMI         : Weight / (Height / 100) ** 2
    Age_log     : log1p(Age)
    Risk_flag   : (Glucose > 126) * 1
    BP_ratio    : Systolic / Diastolic

Columns with spaces or special characters: use col('col name with spaces')

Available math functions: log, log1p, log2, log10, sqrt, exp, abs,
    sin, cos, tan, floor, ceil, clip, where, minimum, maximum,
    isnan, isinf, nan, inf, pi

Column identifiers that are valid Python names can be used directly.
Chained feature references are supported — if BMI depends on Weight and Height,
and another feature depends on BMI, the correct evaluation order is resolved
automatically via topological sort.

Security note
-------------
Expressions are evaluated through a whitelist-based AST validator.
No imports, file I/O, attribute traversal, or builtins are available.
"""
from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------

# Whitelist of allowed AST node types — everything else raises ValueError.
_ALLOWED_AST = frozenset({
    ast.Expression,
    # Operators
    ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Invert, ast.Not,
    ast.And, ast.Or,
    # Comparisons
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    # Literals
    ast.Constant,
    # Names (variables = columns + math functions)
    ast.Name, ast.Load,
    # Function calls and keyword args
    ast.Call, ast.keyword,
    # Ternary: value_if_true if condition else value_if_false
    ast.IfExp,
    # Tuple literal (useful for clip bounds etc.)
    ast.Tuple,
})

# Math functions injected into every expression namespace.
_MATH_NS: dict[str, Any] = {
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "floor": np.floor,
    "ceil": np.ceil,
    "clip": np.clip,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "isnan": np.isnan,
    "isinf": np.isinf,
    "nan": np.nan,
    "inf": np.inf,
    "pi": np.pi,
    "True": True,
    "False": False,
}

# Names explicitly blocked at AST validation time (even though __builtins__={} also prevents them).
_BLOCKED_NAMES = frozenset({
    "exec", "eval", "compile", "open", "input", "print",
    "exit", "quit", "help", "dir", "vars", "locals", "globals",
    "getattr", "setattr", "delattr", "hasattr", "object", "type",
    "breakpoint", "memoryview", "bytearray", "bytes",
})

# Reserved names that cannot be used as column identifiers.
_RESERVED = frozenset(_MATH_NS) | {"col"}


def _validate_ast(expr_str: str) -> ast.Expression:
    """Parse and validate an expression string.  Returns the compiled AST tree."""
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in expression: {exc}") from exc

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST:
            raise ValueError(
                f"Disallowed construct '{type(node).__name__}' in expression. "
                "Only arithmetic, comparison, and whitelisted math functions are allowed."
            )
        # Block dunder names (__import__, __builtins__, __class__, etc.)
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(
                f"Disallowed name '{node.id}' in expression. "
                "Names starting with '__' are not allowed."
            )
        # Block explicitly dangerous builtins by name.
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise ValueError(
                f"Disallowed function '{node.id}' in expression."
            )
    return tree


def _eval_expression(
    expr_str: str,
    df: pd.DataFrame,
    tree: ast.Expression | None = None,
) -> pd.Series:
    """
    Evaluate *expr_str* against *df* and return a pd.Series aligned to df.index.

    Parameters
    ----------
    expr_str : str
        The user expression, e.g. ``"Weight / (Height / 100) ** 2"``.
    df : pd.DataFrame
        DataFrame providing column values.
    tree : ast.Expression, optional
        Pre-parsed/validated AST tree (avoids re-parsing in hot loops).
    """
    if tree is None:
        tree = _validate_ast(expr_str)

    # Build namespace: math functions + each column as a float64 Series
    ns: dict[str, Any] = dict(_MATH_NS)
    ns["__builtins__"] = {}

    for col_name in df.columns:
        # Only expose identifier-safe names directly; others via col()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col_name) and col_name not in _RESERVED:
            ns[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    def _col_accessor(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    ns["col"] = _col_accessor

    code = compile(tree, "<fe_expression>", "eval")
    result = eval(code, ns)  # noqa: S307 — controlled namespace, AST validated

    # Coerce result to a float Series aligned to df.index
    if isinstance(result, pd.Series):
        return result.reset_index(drop=True).set_axis(df.index)
    if isinstance(result, np.ndarray):
        return pd.Series(result, index=df.index, dtype=float)
    # Scalar — broadcast
    return pd.Series(float(result), index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Topological sort (for FE features referencing other FE features)
# ---------------------------------------------------------------------------

def _extract_name_refs(expr_str: str) -> set[str]:
    """Return all ast.Name identifiers referenced in *expr_str*."""
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _topo_sort(defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return *defs* in evaluation order such that if feature B's expression
    references feature A's name, A appears before B.
    Raises ValueError on circular dependencies.
    """
    name_to_def = {d["name"]: d for d in defs}
    all_fe_names = set(name_to_def)
    order: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(
                f"Feature engineering: circular dependency detected involving '{name}'"
            )
        if name not in order:
            visiting.add(name)
            refs = _extract_name_refs(name_to_def[name].get("expression", ""))
            for ref in refs:
                if ref in all_fe_names:
                    visit(ref)
            visiting.discard(name)
            order.append(name)

    for n in list(all_fe_names):
        visit(n)

    return [name_to_def[n] for n in order]


# ---------------------------------------------------------------------------
# FeatureEngineeringTransformer
# ---------------------------------------------------------------------------

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    feature_defs : list of dicts, each with keys:
        name       : str   — output column name
        expression : str   — Python-like formula, e.g. ``"Weight / (Height/100)**2"``
        enabled    : bool  — if False, the feature is skipped (default True)
    """

    def __init__(self, feature_defs: list[dict[str, Any]] | None = None):
        self.feature_defs = feature_defs or []

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineeringTransformer":
        active = [d for d in self.feature_defs if d.get("enabled", True)]
        sorted_defs = _topo_sort(active)

        # Pre-validate and compile all expressions during fit so errors surface early.
        self._sorted_defs_: list[dict[str, Any]] = []
        self._compiled_: dict[str, ast.Expression] = {}

        for fdef in sorted_defs:
            name = fdef["name"]
            expr = str(fdef.get("expression", "")).strip()
            if not expr:
                raise ValueError(
                    f"Feature engineering: feature '{name}' has an empty expression."
                )
            try:
                tree = _validate_ast(expr)
            except ValueError as exc:
                raise ValueError(
                    f"Feature engineering: invalid expression for '{name}': {exc}"
                ) from exc
            self._compiled_[name] = tree
            self._sorted_defs_.append(fdef)

        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        if not hasattr(self, "_sorted_defs_"):
            raise RuntimeError(
                "FeatureEngineeringTransformer must be fitted before calling transform()."
            )
        if not self._sorted_defs_:
            return X.copy()

        result = X.copy()
        for fdef in self._sorted_defs_:
            name = fdef["name"]
            expr = str(fdef.get("expression", "")).strip()
            tree = self._compiled_.get(name)
            try:
                result[name] = _eval_expression(expr, result, tree=tree)
            except Exception as exc:
                raise ValueError(
                    f"Feature engineering: error evaluating '{name}' "
                    f"(expression: {expr!r}): {exc}"
                ) from exc

        return result

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        if not hasattr(self, "_sorted_defs_"):
            return list(input_features or [])
        created = [d["name"] for d in self._sorted_defs_]
        base = list(input_features) if input_features is not None else []
        return base + [n for n in created if n not in base]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_created_feature_names(self) -> list[str]:
        if not hasattr(self, "_sorted_defs_"):
            return []
        return [d["name"] for d in self._sorted_defs_]

    def is_noop(self) -> bool:
        """True when no active feature definitions are present."""
        active = [d for d in self.feature_defs if d.get("enabled", True)]
        return len(active) == 0


# ---------------------------------------------------------------------------
# Validation helper (call before fit to surface config errors early)
# ---------------------------------------------------------------------------

def validate_feature_defs(
    feature_defs: list[dict[str, Any]],
    available_columns: list[str],
) -> list[str]:
    """
    Return a list of human-readable error strings.  Empty list means valid.
    Does NOT raise — caller decides whether to fail or warn.
    """
    errors: list[str] = []
    seen_names: set[str] = set()
    created_names: set[str] = set()

    for fdef in feature_defs:
        if not fdef.get("enabled", True):
            continue

        name = str(fdef.get("name", "")).strip()
        expr = str(fdef.get("expression", "")).strip()

        if not name:
            errors.append("A feature definition is missing a 'name'.")
            continue

        if name in seen_names:
            errors.append(f"Duplicate feature name: '{name}'.")
        seen_names.add(name)

        if name in available_columns:
            errors.append(
                f"Feature '{name}' would overwrite an existing column. "
                "Choose a different name."
            )

        if not expr:
            errors.append(f"Feature '{name}' has an empty expression.")
            created_names.add(name)
            continue

        try:
            _validate_ast(expr)
        except ValueError as exc:
            errors.append(f"Feature '{name}': {exc}")
            created_names.add(name)
            continue

        # Check that all Name references in the expression exist as columns
        # (either original columns or previously defined FE columns).
        reachable = set(available_columns) | created_names | _RESERVED | {"col"}
        refs = _extract_name_refs(expr)
        unknown = refs - reachable
        if unknown:
            errors.append(
                f"Feature '{name}': expression references unknown names: "
                f"{sorted(unknown)}. "
                "These are not column names, FE-defined names, or math functions."
            )

        created_names.add(name)

    return errors
