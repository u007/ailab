from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCATE_", env_file=".env", extra="ignore")

    model_id: str = "nvidia/LocateAnything-3B"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "auto"
    device_map: str | None = None
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    max_new_tokens: int = Field(default=2048, ge=1)
    max_concurrency: int = Field(default=2, ge=1)
    shutdown_grace_seconds: int = Field(default=30, ge=0)
    min_image_pixels: int = Field(default=0, ge=0)
    max_image_pixels: int = Field(default=1_048_576, ge=0)
    max_image_side: int = Field(default=1536, ge=0)
    generation_mode: str = "hybrid"
    top_p: float = Field(default=0.9, ge=0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0)
    verbose_generation: bool = True
    preload_model: bool = False
    log_responses: bool = True
    log_prompts: bool = True
    prompt_mode: Literal["preserve", "template"] = "preserve"
    empty_cuda_cache_after_generate: bool = True
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
