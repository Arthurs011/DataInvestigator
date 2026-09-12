"""Runtime configuration from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    model: str = field(default_factory=lambda: os.getenv("INVESTIGATOR_MODEL", "openai/gpt-4o-mini"))
    row_limit: int = field(default_factory=lambda: int(os.getenv("INVESTIGATOR_ROW_LIMIT", "0")))
    seed: int = 42
    significance_alpha: float = 0.05
    top_findings: int = 10
    max_pr: int = 300  # max chars of prompt "paragraph" per finding


def load_config() -> Config:
    return Config()