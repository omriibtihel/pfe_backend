# Compatibility shim — ColumnAligner moved to app.services.preprocessing.transformers
# Keeps old pickled pipelines loadable after the package reorganization.
from app.services.preprocessing.transformers import ColumnAligner  # noqa: F401
