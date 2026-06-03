from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "MedIQ API"
    ENV: str = "dev"
    SECRET_KEY: str = "super-secret-key"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 h — une session de travail

    DATABASE_URL: str = "postgresql+psycopg2://medicalvision:medicalvision@localhost:5432/medicalvision"

    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Email / SMTP
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@mediq.ai"
    SMTP_FROM_NAME: str = "MedIQ"
    FRONTEND_URL: str = "http://localhost:5173"

    # LLM reporting backends. Empty values disable a backend.
    # Default deployment uses Groq (managed). Ollama is kept available for
    # on-prem / RGPD-strict deployments — set the two URL/MODEL vars in .env
    # to enable it. By default both are empty so Ollama is skipped silently.
    REPORTING_OLLAMA_URL: str = ""
    REPORTING_OLLAMA_MODEL: str = ""
    REPORTING_OLLAMA_TIMEOUT_S: int = 30
    REPORTING_GROQ_API_KEY: str = ""
    REPORTING_GROQ_MODEL: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        extra = "ignore"


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

STORAGE_PATH = BASE_DIR / "storage"
PROJECTS_PATH = STORAGE_PATH / "projects"

