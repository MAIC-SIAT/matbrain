"""Runtime configuration loaded from .env / environment via pydantic-settings.

Each model role (T1 executor / R1 reasoner) can independently target a
compatible chat-completion provider and endpoint.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parent.parent / ".env",
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- shared fallback endpoint (used when a role leaves its own blank) ----
    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_base_url: str = Field("", alias="LLM_BASE_URL")

    # ---- Mat-T1 (executor) ----
    mat_t1_provider: str = Field("openai-compatible", alias="MAT_T1_PROVIDER")
    mat_t1_base_url: str = Field("", alias="MAT_T1_BASE_URL")
    mat_t1_api_key: str = Field("", alias="MAT_T1_API_KEY")
    mat_t1_model: str = Field("", alias="MAT_T1_MODEL")

    # ---- Mat-R1 (reasoner) ----
    mat_r1_provider: str = Field("openai-compatible", alias="MAT_R1_PROVIDER")
    mat_r1_base_url: str = Field("", alias="MAT_R1_BASE_URL")
    mat_r1_api_key: str = Field("", alias="MAT_R1_API_KEY")
    mat_r1_model: str = Field("", alias="MAT_R1_MODEL")

    # ---- MCP endpoints ----
    mcp_mat_query_url: str = Field("http://localhost:5667/sse", alias="MCP_MAT_QUERY_URL")
    mcp_matgl_url: str = Field("http://localhost:5668/sse", alias="MCP_MATGL_URL")
    mcp_crystallm_url: str = Field("http://localhost:5669/sse", alias="MCP_CRYSTALLM_URL")
    mcp_mattergen_url: str = Field("http://localhost:5670/sse", alias="MCP_MATTERGEN_URL")
    mcp_pymatgen_url: str = Field("http://localhost:5672/pymatgen/sse", alias="MCP_PYMATGEN_URL")
    mcp_smact_url: str = Field("http://localhost:5673/sse", alias="MCP_SMACT_URL")
    mcp_pyxtal_url: str = Field("http://localhost:5675/sse", alias="MCP_PYXTAL_URL")
    mcp_verifier_url: str = Field("http://localhost:5676/sse", alias="MCP_VERIFIER_URL")
    mcp_matminer_url: str = Field("http://localhost:5674/sse", alias="MCP_MATMINER_URL")

    # ---- agent runtime ----
    max_iterations: int = Field(6, alias="MAX_ITERATIONS")
    tool_call_timeout: int = Field(480, alias="TOOL_CALL_TIMEOUT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def _fill_role_defaults(self) -> "Settings":
        """If per-role endpoint settings are blank, fall back to shared values."""
        if not self.mat_t1_base_url:
            self.mat_t1_base_url = self.llm_base_url
        if not self.mat_t1_api_key:
            self.mat_t1_api_key = self.llm_api_key
        if not self.mat_r1_base_url:
            self.mat_r1_base_url = self.llm_base_url
        if not self.mat_r1_api_key:
            self.mat_r1_api_key = self.llm_api_key
        return self

    def mcp_endpoints(self) -> dict[str, str]:
        # MatterGen is not wired into the agent by default. To enable it, add
        # "mattergen": self.mcp_mattergen_url to the mapping below.
        return {
            "mat-query": self.mcp_mat_query_url,
            "matgl": self.mcp_matgl_url,
            "crystallm": self.mcp_crystallm_url,
            "pymatgen": self.mcp_pymatgen_url,
            "smact": self.mcp_smact_url,
            "pyxtal": self.mcp_pyxtal_url,
            "verifier": self.mcp_verifier_url,
            "matminer": self.mcp_matminer_url,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
