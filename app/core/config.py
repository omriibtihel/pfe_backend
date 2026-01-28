from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "MedicalVision API"
    ENV: str = "dev"
    SECRET_KEY: str = "super-secret-key"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    DATABASE_URL: str = "postgresql+psycopg2://medicalvision:medicalvision@localhost:5432/medicalvision"

    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"  


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

STORAGE_PATH = BASE_DIR / "storage"
PROJECTS_PATH = STORAGE_PATH / "projects"

