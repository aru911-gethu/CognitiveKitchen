"""Runtime configuration, loaded from .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"

    api_host: str = "127.0.0.1"
    api_port: int = 8010   # 8000 is taken by AWSWorkDocsDriveClient on this machine

    data_dir: Path = ROOT / "data"
    upload_dir: Path = ROOT / "data" / "uploads"
    ingested_dir: Path = ROOT / "data" / "ingested"

    # models - single choice each, overridden by .env on a GPU machine
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    generation_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reranker_model: str = "BAAI/bge-reranker-base"
    judge_model: str = "gpt-4o-mini"

    # knowledge graph -- Neo4j Aura. On Aura the username and the database are
    # both the instance id (e.g. "09bfbafe"), NOT the literal "neo4j".
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None
    aura_instanceid: str | None = None
    aura_instancename: str | None = None

    # observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "cognitive-kitchen"
    openai_api_key: str | None = None

    # sample sizes while proving the pipeline
    eval_retrieval_questions: int = 20
    eval_generation_questions: int = 5

    # scraping guardrails
    max_pages_per_run: int = 60
    request_timeout_ms: int = 30_000
    politeness_delay_ms: int = 150

    @property
    def graph_configured(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_username and self.neo4j_password)

    @property
    def api_base(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.ingested_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
