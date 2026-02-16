from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.auth import router as auth_router
from app.api.routes.admin import router as admin_router
from app.api.routes.projects import router as projects_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.database import router as database_router
from app.api.routes.charts import router as charts_router
from app.api.routes.processing import router as processing_router
from app.api.routes.versions import router as versions_router
from app.api.routes.training import router as training_router
from app.api.routes.versions_workspace import router as versions_workspace_router
from app.api.routes.version_schema import router as version_schema_router





app = FastAPI(title=settings.APP_NAME)

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
app.include_router(projects_router, prefix="/api/projects", tags=["Projects"])
app.include_router(datasets_router, prefix="/api/projects/{project_id}/datasets", tags=["Datasets"])
app.include_router(database_router, prefix="/api/projects/{project_id}", tags=["Database"])
app.include_router(charts_router,prefix="/api/projects/{project_id}/datasets",tags=["Charts"])
app.include_router(processing_router, prefix="/api/projects/{project_id}", tags=["processing"])
app.include_router(versions_router, prefix="/api/projects/{project_id}/versions", tags=["versions"])
app.include_router(training_router, prefix="/api/projects/{project_id}/training", tags=["training"])
app.include_router(versions_workspace_router,prefix="/api/projects/{project_id}/versions",tags=["versions-workspace"],)
app.include_router(version_schema_router,prefix="/api/projects/{project_id}/versions",tags=["versions-schema"],)




